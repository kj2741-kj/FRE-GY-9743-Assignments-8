"""IFR components, calibration Jacobian, and market-data labels from the template."""

import numpy as np
import QuantLib as ql
from scipy.linalg import block_diag
from fixedincomelib.valuation import (
    FundingIndexParameter, ValuationParametersCollection,
    ValuationEngineProductRegistry, ValuationRequest,
)
from typing import List, Optional

from fixedincomelib.date import Date, accrued
from fixedincomelib.data import DataCollection
from fixedincomelib.model import (
    Model, ModelComponent, ModelType, BuildMethodCollection,
    ModelBuilderRegistry, ModelDeserializerRegistry,
)
from fixedincomelib.product import Product
from fixedincomelib.utilities import Interpolator1D, InterpolatorFactory
from fixedincomelib.yield_curve.build_method import YieldCurveIndexBuildMethod, YieldCurveFundingBuildMethod

__all__ = ["YieldCurve", "YieldCurveModelComponent"]


class YieldCurve(Model):

    _version = 1
    _model_type = ModelType.YIELD_CURVE

    def __init__(
        self,
        value_date: Date,
        data_collection: DataCollection,
        build_method_collection: BuildMethodCollection,
    ) -> None:
        super().__init__(value_date, self._model_type, data_collection, build_method_collection)

    def serialize(self) -> dict:
        content = {}
        content["VERSION"] = YieldCurve._version
        content["MODEL_TYPE"] = YieldCurve._model_type.to_string()
        content["VALUE_DATE"] = self.value_date.ISO()
        content["BUILD_METHOD_COLLECTION"] = self.build_method_collection.serialize()
        content["DATA_COLLECTION"] = self.data_collection.serialize()
        return content

    @classmethod
    def deserialize(cls, input_dict: dict) -> "YieldCurve":
        input_dict_ = input_dict.copy()
        assert "VERSION" in input_dict_
        version = input_dict_["VERSION"]
        assert "MODEL_TYPE" in input_dict_
        model_type = input_dict_["MODEL_TYPE"]
        assert "VALUE_DATE" in input_dict_
        value_date = Date(input_dict_["VALUE_DATE"])
        bmc = BuildMethodCollection.deserialize(input_dict_["BUILD_METHOD_COLLECTION"])
        dc = DataCollection.deserialize(input_dict_["DATA_COLLECTION"])
        # find modelbuilder
        func = ModelBuilderRegistry().get(model_type)
        return func(value_date, dc, bmc)

    def discount_factor(
        self, index: ql.Index | str, expiry_date: Date, payment_currency: Optional[str] = None
    ):
        """Combine local discount factors recursively through component references.

        ``payment_currency`` is reserved for future use and does not affect the
        calculation. The reference is read from each stored component identifier.
        """
        visited = set()

        def discount_factor_with_reference(component_index):
            this_component: YieldCurveModelComponent = self.retrieve_model_component(
                component_index
            )
            this_index = this_component.component_identifier
            component_name = this_index.name()
            if component_name in visited:
                raise ValueError(f"Circular yield-curve reference: {component_name}")
            visited.add(component_name)

            df = 1.0
            reference_index = getattr(this_index, "reference_index", None)
            if reference_index is not None:
                df = discount_factor_with_reference(reference_index)
            return df * this_component.discount_factor(expiry_date)

        return discount_factor_with_reference(index)

    def discount_factor_gradient_wrt_state(
        self, index: ql.Index | str, expiry_date: Date,
        gradient_vector: List[np.ndarray],
        scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write the recursive DF gradient, in component creation order.

        For D = D_local * D_reference, apply the product rule along the
        same reference chain as ``discount_factor``. Work in a temporary
        gradient so recursion never rescales the caller's existing values.
        The reference chain is assumed valid and acyclic.
        """
        contribution = []
        self.resize_gradient(contribution)

        def gradient_with_reference(component_index):
            component = self.retrieve_model_component(component_index)
            identifier = component.component_identifier

            reference = getattr(identifier, "reference_index", None)
            reference_df = 1.0 if reference is None else gradient_with_reference(reference)
            local_df = component.discount_factor(expiry_date)
            # D_local * grad(D_reference).
            for block in contribution:
                block *= local_df
            # D_reference * grad(D_local), in this component's own block.
            position = self.retrieve_model_component_index(identifier)
            component.discount_factor_gradient_wrt_state(
                expiry_date, contribution[position], scaler=reference_df, accumulate=True,
            )
            return local_df * reference_df

        gradient_with_reference(index)
        self.resize_gradient(gradient_vector)
        for output, block in zip(gradient_vector, contribution):
            if accumulate:
                output += scaler * block
            else:
                output[:] = scaler * block

    def calculate_model_jacobian(self):
        if self.is_jacobian_calculated_:
            return
        # WARNING: WE DO NOT ALLOW A MIXTURE OF CALIBRATION INSTRUMENTS AND STATE DATA FOR NOW
        only_state_data = False
        jacobian_pre = [None] * self.num_components
        for target_name, yc_component in self.components_.items():
            index = self.component_indices[target_name]
            calib_prod = yc_component.calibration_product
            calib_funding = yc_component.calibration_funding
            if len(calib_prod) == 0:
                # no calibration, just using state data
                # jacobian is identity
                jacobian_pre[index] = np.diag(np.ones(yc_component.num_state_data))
                only_state_data = True
                continue
            # calculate calibration instrument gradient
            grads = []
            for _, (prod, funding) in enumerate(zip(calib_prod, calib_funding)):
                fi_vp = FundingIndexParameter({"Funding Index": funding})
                vpc = ValuationParametersCollection([fi_vp])
                engine = ValuationEngineProductRegistry.new_valuation_engine(
                    self, prod, vpc, ValuationRequest.PV_DETAILED
                )
                engine.calculate_value()
                # One Jacobian row: d(par rate or spread) / dX^I, with the gradient blocks
                # flattened in model component order.
                grads.append(np.concatenate(engine.grad_at_par(), axis=0))
            jacobian_pre[index] = grads  # np.concatenate(grads, axis=0)

        self.model_jacobian_ = (
            block_diag(*jacobian_pre) if only_state_data else np.concatenate(jacobian_pre, axis=0)
        )
        self.is_jacobian_calculated_ = True

    def risk_postprocess(self, grad: np.ndarray):
        frame = [None] * self.num_components
        for target_name, yc_component in self.components_.items():
            index = self.component_indices[target_name]
            frame[index] = yc_component.market_data
        frame = np.concatenate(frame, axis=0)
        return np.concatenate([frame, grad.reshape(len(frame), 1)], axis=1)


class YieldCurveModelComponent(ModelComponent):

    def __init__(
        self,
        value_date: Date,
        component_identifier: ql.Index,
        state_data: np.ndarray,
        build_method: YieldCurveIndexBuildMethod | YieldCurveFundingBuildMethod,
        calibration_product: Optional[List[Product]] = None,
        calibration_funding: Optional[List[str]] = None,
        market_data: Optional[List] = None,
    ) -> None:

        super().__init__(
            value_date,
            component_identifier,
            state_data,
            build_method,
            [] if calibration_product is None else calibration_product,
            [] if calibration_funding is None else calibration_funding,
            [] if market_data is None else market_data,
        )
        assert len(state_data) == 2
        self.num_state_data_ = len(state_data[0])
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            state_data[0],
            state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method,
        )

    def perturb_model_parameter(
        self, parameter_id: int, perturb_size: float, override_parameter: Optional[bool] = False
    ):
        super().perturb_model_parameter(parameter_id, perturb_size, override_parameter)
        self.interpolator_ = InterpolatorFactory.create_1d_interpolator(
            self.state_data[0],
            self.state_data[1],
            self.build_method.interpolation_method,
            self.build_method.extrapolation_method,
        )

    @property
    def state_data_interpolator(self) -> Interpolator1D:
        return self.interpolator_

    @property
    def num_state_data(self) -> int:
        return self.num_state_data_

    def discount_factor(self, expiry_date: Date):
        """Discount using this component's IFR only, excluding any reference curve."""
        if expiry_date.serialNumber() == 0:
            raise ValueError("A valid discount-factor date is required.")
        if expiry_date < self.value_date:
            raise ValueError("Discount-factor date cannot precede the model value date.")

        # Use the same time convention as the IFR node construction.
        time_to_expiry = accrued(self.value_date, expiry_date)
        exponent = self.state_data_interpolator.integrate(0.0, time_to_expiry)
        return np.exp(-exponent)

    def discount_factor_gradient_wrt_state(
        self, expiry_date: Date, gradient_vector: np.ndarray,
        scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        """Write -D(T) * grad(integral_0^T IFR(t) dt) for this component.

        The caller supplies a writable floating-point vector of the node
        count. ``scaler`` multiplies only this call's contribution, in raw
        decimal-rate units; basis-point scaling belongs to report display.
        """
        df = self.discount_factor(expiry_date)
        time_to_expiry = accrued(self.value_date, expiry_date)
        weights = self.state_data_interpolator.gradient_of_integrated_value_wrt_ordinate(
            0.0, time_to_expiry,
        )
        contribution = -scaler * df * weights
        if accumulate:
            gradient_vector += contribution
        else:
            gradient_vector[:] = contribution


ModelDeserializerRegistry().register(YieldCurve._model_type.to_string(), YieldCurve.deserialize)
