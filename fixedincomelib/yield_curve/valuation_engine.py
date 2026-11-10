"""Analytic cashflow, stream and composite-product valuation on the IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


# Stream and composite engines follow the instructor template's decomposition.
from fixedincomelib.date import TermOrDate
from fixedincomelib.product import (
    ProductForwardRateCashflow, ProductOvernightIndexCashflow, ProductInterestRateStream,
    ProductFRA, ProductRFRSwap, ProductOvernightIndexBasisSwap, PayOrReceive,
)
from fixedincomelib.valuation.report import CashflowsReport
from fixedincomelib.yield_curve.valuation_engine_analytics import (
    ValuationEngineAnalyticsForwardRate, ValuationEngineAnalyticsOvernightIndex,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type,
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)

    def calculate_value(self) -> None:
        product = self.product_
        self.value_ = self.cash_ = 0.0
        if product.payment_date < self.value_date:
            return
        # Notional already carries the pay/receive sign; accrued is a fraction.
        amount = product.notional * product.accrued
        if product.payment_date == self.value_date:
            self.value_ = self.cash_ = amount
        else:
            self.value_ = amount * self.model.discount_factor(self.funding_index_, product.payment_date)

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        product = self.product_
        if product.payment_date <= self.value_date:
            self.model.resize_gradient(gradient)
            if not accumulate:
                for block in gradient:
                    block.fill(0.0)
            return
        self.model.discount_factor_gradient_wrt_state(
            self.funding_index_, product.payment_date, gradient,
            scaler=scaler * product.notional * product.accrued,
            accumulate=accumulate,
        )


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)


class ValuationEngineInterestRateStream(ValuationEngineProduct):
    """Value each coupon on its payment date, using index and funding curves."""

    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.stream_ = product
        funding = valuation_parameters_collection.get_vp_from_build_method_collection(FundingIndexParameter._vp_type)
        self.funding_index_ = funding.get_funding_index(product.currency)
        self.index_engines_ = []
        for i in range(product.num_cashflows()):
            cf = product.cashflow(i)
            if isinstance(cf, ProductForwardRateCashflow):
                engine = ValuationEngineAnalyticsForwardRate(
                    model, valuation_parameters_collection, cf.projection_index,
                    cf.effective_date, cf.termination_date, cf.accrual_basis)
            elif isinstance(cf, ProductOvernightIndexCashflow):
                engine = ValuationEngineAnalyticsOvernightIndex(
                    model, valuation_parameters_collection, cf.on_index,
                    cf.effective_date, TermOrDate(cf.termination_date), cf.compounding_method)
            else:
                engine = None
            self.index_engines_.append(engine)

    @staticmethod
    def _accrual(cf):
        if isinstance(cf, ProductOvernightIndexCashflow):
            return cf.on_index.dayCounter().yearFraction(cf.effective_date, cf.termination_date)
        return cf.accrued

    def calculate_value(self):
        self.value_ = self.cash_ = 0.0
        n = self.stream_.num_cashflows()
        self.payoffs_ = [0.0] * n
        self.dfs_ = [0.0] * n
        self.rates_ = [None] * n
        self.accruals_ = [self._accrual(self.stream_.cashflow(i)) for i in range(n)]
        for i in range(n):
            cf = self.stream_.cashflow(i)
            if cf.payment_date < self.value_date:
                continue
            index_engine = self.index_engines_[i]
            if index_engine is None:
                rate = self.stream_.fixed_rate
            else:
                index_engine.calculate_value()
                rate = index_engine.value()
            spread = cf.spread if isinstance(cf, ProductOvernightIndexCashflow) else 0.0
            self.rates_[i] = rate
            self.payoffs_[i] = cf.notional * self.accruals_[i] * (rate + spread)
            df = 1.0 if cf.payment_date == self.value_date else self.model.discount_factor(self.funding_index_, cf.payment_date)
            self.dfs_[i] = df
            self.value_ += self.payoffs_[i] * df
            if cf.payment_date == self.value_date:
                self.cash_ += self.payoffs_[i]

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        if any(isinstance(helper, ValuationEngineAnalyticsForwardRate) for helper in self.index_engines_):
            raise NotImplementedError("Term-stream risk is outside the OIS assignment.")
        self.calculate_value()
        self.model.resize_gradient(gradient)
        if not accumulate:
            for block in gradient:
                block.fill(0.0)
        for i in range(self.stream_.num_cashflows()):
            cf = self.stream_.cashflow(i)
            if cf.payment_date < self.value_date:
                continue
            if cf.payment_date > self.value_date:
                self.model.discount_factor_gradient_wrt_state(
                    self.funding_index_, cf.payment_date, gradient,
                    scaler=scaler * self.payoffs_[i], accumulate=True)
            if self.index_engines_[i] is not None:
                self.index_engines_[i].calculate_risk(
                    gradient, scaler=scaler * cf.notional * self.accruals_[i] * self.dfs_[i],
                    accumulate=True)


    def create_cash_flows_report(self):
        self.calculate_value()
        rows = []
        for i in range(self.stream_.num_cashflows()):
            cf = self.stream_.cashflow(i)
            rows.append({
                "Leg": "Stream", "Currency": cf.currency.code(),
                "Accrual Start": cf.effective_date.ISO(), "Accrual End": cf.termination_date.ISO(),
                "Payment Date": cf.payment_date.ISO(), "Notional": cf.notional,
                "Accrual": self.accruals_[i], "Rate": self.rates_[i],
                "Spread": cf.spread if isinstance(cf, ProductOvernightIndexCashflow) else 0.0,
                "Cashflow": self.payoffs_[i], "Discount Factor": self.dfs_[i],
                "PV": self.payoffs_[i] * self.dfs_[i],
            })
        return CashflowsReport(rows)


def _combine_cashflow_reports(legs):
    rows = []
    for name, sign, engine in legs:
        for original in engine.create_cash_flows_report().content:
            row = dict(original, Leg=name)
            for key in ("Notional", "Cashflow", "PV"):
                row[key] *= sign
            rows.append(row)
    return CashflowsReport(rows)


class ValuationEngineProductRfrSwap(ValuationEngineProduct):
    """Fixed and overnight-floating leg valuation."""

    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.fixed_leg_sign_ = -1.0 if product.pay_or_rec == PayOrReceive.PAY else 1.0
        self.floating_leg_sign_ = -self.fixed_leg_sign_
        self.fixed_leg_engine_ = ValuationEngineInterestRateStream(model, valuation_parameters_collection, product.fixed_leg, request)
        self.floating_leg_engine_ = ValuationEngineInterestRateStream(model, valuation_parameters_collection, product.floating_leg, request)
        self.fixed_rate_ = product.fixed_rate
        self.notional_ = product.notional
        self.fixed_value_ = self.float_value_ = 0.0
        self.fixed_cash_ = self.float_cash_ = 0.0
        self.annuity_ = self.par_rate_or_spread_ = 0.0


    def calculate_value(self):

        self.fixed_leg_engine_.calculate_value()
        self.floating_leg_engine_.calculate_value()
        self.fixed_value_ = self.fixed_leg_engine_.value_
        self.float_value_ = self.floating_leg_engine_.value_
        self.fixed_cash_ = self.fixed_leg_engine_.cash_
        self.float_cash_ = self.floating_leg_engine_.cash_

        self.value_ = (
            self.fixed_leg_sign_ * self.fixed_value_ + self.floating_leg_sign_ * self.float_value_
        )
        self.cash_ = (
            self.fixed_leg_sign_ * self.fixed_cash_ + self.floating_leg_sign_ * self.float_cash_
        )
        # The fixed-leg PV is notional * fixed_rate * annuity, so dividing by the nonzero
        # coupon and the notional leaves the annuity per unit notional.
        self.annuity_ = self.fixed_value_ / (self.fixed_rate_ * self.notional_)
        # The par rate is the coupon that equates the fixed-leg and floating-leg PVs.
        self.par_rate_or_spread_ = self.float_value_ / (self.annuity_ * self.notional_)

    def calculate_first_order_risk(
        self, gradient=None, scaler: float = 1.0, accumulate: bool = False
    ) -> None:

        local_grad = []
        self.model_.resize_gradient(local_grad)

        self.fixed_leg_engine_.calculate_first_order_risk(
            local_grad, scaler=self.fixed_leg_sign_ * scaler, accumulate=True
        )
        self.floating_leg_engine_.calculate_first_order_risk(
            local_grad, scaler=self.floating_leg_sign_ * scaler, accumulate=True
        )

        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def par_rate_or_spread(self) -> float:
        return self.par_rate_or_spread_

    def grad_at_par(self) -> np.ndarray:

        local_grad = []
        self.model_.resize_gradient(local_grad)

        # Par rate P = V_float / (N * A) with A = V_fixed / (N * K), so V_fixed = N * K * A.
        # Quotient rule with the coupon K held constant:
        #   dP = dV_float / (N * A) - (P / A) * dA,   dA = dV_fixed / (N * K).
        self.calculate_value()
        notional_annuity = self.notional_ * self.annuity_
        self.floating_leg_engine_.calculate_first_order_risk(
            local_grad, scaler=1.0 / notional_annuity, accumulate=True
        )
        self.fixed_leg_engine_.calculate_first_order_risk(
            local_grad,
            scaler=-self.par_rate_or_spread_ / (notional_annuity * self.fixed_rate_),
            accumulate=True,
        )

        return local_grad

    def create_cash_flows_report(self):
        return _combine_cashflow_reports([
            ("Fixed", self.fixed_leg_sign_, self.fixed_leg_engine_),
            ("Floating", self.floating_leg_sign_, self.floating_leg_engine_),
        ])


class ValuationEngineProductFra(ValuationEngineProduct):
    """End-paid FRA valued as a fixed leg minus a term-floating leg."""

    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.fixed_leg_sign_ = -1.0 if product.pay_or_rec == PayOrReceive.PAY else 1.0
        self.floating_leg_sign_ = -self.fixed_leg_sign_
        self.fixed_leg_engine_ = ValuationEngineInterestRateStream(model, valuation_parameters_collection, product.fixed_leg, request)
        self.floating_leg_engine_ = ValuationEngineInterestRateStream(model, valuation_parameters_collection, product.floating_leg, request)

    def calculate_value(self):
        self.fixed_leg_engine_.calculate_value()
        self.floating_leg_engine_.calculate_value()
        self.value_ = (self.fixed_leg_sign_ * self.fixed_leg_engine_.value +
                       self.floating_leg_sign_ * self.floating_leg_engine_.value)
        self.cash_ = (self.fixed_leg_sign_ * self.fixed_leg_engine_.cash +
                      self.floating_leg_sign_ * self.floating_leg_engine_.cash)

    def create_cash_flows_report(self):
        return _combine_cashflow_reports([
            ("Fixed", self.fixed_leg_sign_, self.fixed_leg_engine_),
            ("Floating", self.floating_leg_sign_, self.floating_leg_engine_),
        ])


class ValuationEngineProductOvernightIndexBasisSwap(ValuationEngineProduct):
    """Template decomposition: leg-1 index, leg-1 spread, and leg-2 index."""

    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.floating_leg_1_sign_ = -1.0 if product.pay_or_rec == PayOrReceive.PAY else 1.0
        self.floating_leg_2_sign_ = -self.floating_leg_1_sign_
        self.floating_leg_1_basis_engine_ = ValuationEngineInterestRateStream(
            model, valuation_parameters_collection, product.floating_leg_1_basis, request)
        self.floating_leg_1_wo_basis_engine_ = ValuationEngineInterestRateStream(
            model, valuation_parameters_collection, product.floating_leg_1_wo_basis, request)
        self.floating_leg_2_engine_ = ValuationEngineInterestRateStream(
            model, valuation_parameters_collection, product.floating_leg_2, request)

    def _legs(self):
        return [("Leg 1 index", self.floating_leg_1_sign_, self.floating_leg_1_wo_basis_engine_),
                ("Leg 1 spread", self.floating_leg_1_sign_, self.floating_leg_1_basis_engine_),
                ("Leg 2 index", self.floating_leg_2_sign_, self.floating_leg_2_engine_)]

    def calculate_value(self):
        self.value_ = self.cash_ = 0.0
        for _, sign, engine in self._legs():
            engine.calculate_value()
            self.value_ += sign * engine.value
            self.cash_ += sign * engine.cash

    def create_cash_flows_report(self):
        return _combine_cashflow_reports(self._legs())


for _product, _engine in (
    (ProductInterestRateStream, ValuationEngineInterestRateStream),
    (ProductFRA, ValuationEngineProductFra),
    (ProductRFRSwap, ValuationEngineProductRfrSwap),
    (ProductOvernightIndexBasisSwap, ValuationEngineProductOvernightIndexBasisSwap),
):
    ValuationEngineProductRegistry().register(
        (ModelType.YIELD_CURVE.to_string(), _product._product_type, AnalyticValParam._vp_type), _engine)

from fixedincomelib.date import accrued
from fixedincomelib.market.registries import FundingIdentifier
from fixedincomelib.product import ProductZeroSpread, LongOrShort
from fixedincomelib.valuation import ValuationParametersCollection, ValuationRequest
from fixedincomelib.valuation.report import PVCashReport
from fixedincomelib.yield_curve.yield_curve_model import YieldCurve

class ValuationEngineProductZeroSpread(ValuationEngineProduct):

    def __init__(
        self,
        model: YieldCurve,
        valuation_parameters_collection: ValuationParametersCollection,
        product: ProductZeroSpread,
        request: ValuationRequest,
    ):
        super().__init__(model, valuation_parameters_collection, product, request)

        # get info from product
        self.currency_ = product.currency
        self.termination_date_ = product.termination_date
        self.sign_ = 1.0 if product.long_or_short == LongOrShort.LONG else -1.0
        self.notional_ = product.notional
        self.index_ = product.index
        self.zero_spread_ = product.zero_rate
        self.time_to_expiry_ = accrued(self.value_date, self.termination_date_)
        # resolve valuation parameters
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_: FundingIdentifier = self.funding_vp_.get_funding_index(self.currency_)

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        if self.value_date <= self.termination_date_:
            funding_model: YieldCurve = self.model_

            self.df_funding_ = funding_model.discount_factor(
                self.funding_index_, self.termination_date_
            )
            self.df_base_ = funding_model.discount_factor(self.index_, self.termination_date_)
            self.value_ = (
                self.sign_
                * self.notional_
                * (
                    self.df_funding_ / self.df_base_
                    - np.exp(-self.zero_spread_ * self.time_to_expiry_)
                )
            )
        else:
            raise Exception("Zero spread product has to be outright.")

    def calculate_first_order_risk(
        self, gradient=None, scaler: float = 1.0, accumulate: bool = False
    ) -> None:

        local_grad = []
        self.model_.resize_gradient(local_grad)

        if self.value_date <= self.termination_date_:
            funding_model: YieldCurve = self.model_
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.termination_date_,
                local_grad,
                scaler * self.sign_ * self.notional_ / self.df_base_,
                True,
            )
            funding_model.discount_factor_gradient_wrt_state(
                self.index_,
                self.termination_date_,
                local_grad,
                -scaler
                * self.sign_
                * self.notional_
                * self.df_funding_
                / self.df_base_
                / self.df_base_,
                True,
            )

        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        raise Exception("ProductZeroSpread does not support cahslfow report")

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def grad_at_par(self):
        local_grad = []
        self.model_.resize_gradient(local_grad)

        if self.value_date <= self.termination_date_:
            # self.value_ = self.sign_ * self.notional_ * ( \
            #     self.df_funding_ / self.df_base_ -  np.exp(-self.zero_spread_ * self.time_to_expiry_))
            # -1/T * ln(df_f / df_base) = s
            df_ratio = self.df_funding_ / self.df_base_
            funding_model: YieldCurve = self.model_
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.termination_date_,
                local_grad,
                -1.0 / self.time_to_expiry_ * df_ratio / self.df_base_,
                True,
            )
            funding_model.discount_factor_gradient_wrt_state(
                self.index_,
                self.termination_date_,
                local_grad,
                1.0
                / self.time_to_expiry_
                * df_ratio
                * self.df_funding_
                / self.df_base_
                / self.df_base_,
                True,
            )

        return local_grad


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductZeroSpread._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductZeroSpread)
