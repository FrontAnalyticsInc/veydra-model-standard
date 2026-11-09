"""Tests for run_multi_scenario intervention gating and calibration.

The test model builds SimulationContext the same way generated models do
(no gating kwargs), so these tests cover existing models in the field.
"""

from datetime import datetime

import pytest

from veydra_model_standard import SimulationContext, VeydraModelStandard

VARIABLES = {
    'simulation.duration': {'default': 10.0},
    'simulation.time_step': {'default': 1.0},
    'm.rate': {'default': 1.0, 'type': 'slider', 'min': 0.0, 'max': 100.0},
    'm.other': {'default': 0.0},
}


class StepModel(VeydraModelStandard):
    """Euler-integrated stock x with dx/dt = rate + other.

    Records the parameter values seen at each step via SimulationContext.get_param.
    """

    @classmethod
    def auto_discover_variables(cls):
        return dict(VARIABLES)

    def __init__(self, params=None):
        super().__init__({})
        self.all_variables = self.auto_discover_variables()
        self.update_parameters(params or {})

    def update_parameters(self, new_params):
        self.params = self._resolve_parameters(self._clean_params(new_params), self.all_variables)

    def run_simulation(self, params=None):
        if params is not None:
            self.update_parameters(params)
        duration = float(self.params['simulation.duration'])
        dt = float(self.params['simulation.time_step'])
        x, t = 0.0, 0.0
        times, xs = [], []
        self.seen = {'m.rate': [], 'm.other': []}
        while t < duration:
            ctx = SimulationContext(
                current_time=t,
                current_datetime=datetime(2024, 1, 1),
                all_params=dict(self.params),
                all_stocks={'m.x': x},
                intervention_active=False,
            )
            rate = ctx.get_param('m.rate')
            other = ctx.get_param('m.other')
            self.seen['m.rate'].append(rate)
            self.seen['m.other'].append(other)
            times.append(t)
            xs.append(x)
            x += (rate + other) * dt
            t += dt
        return {'success': True, 'time': times, 'stocks': {'m.x': xs}, 'flows': {}}


def run(config):
    """Run a multi-scenario config and also return per-scenario seen params."""
    model = StepModel()
    seen = {}
    original = model.run_simulation

    def recording_run(params=None):
        result = original(params)
        seen[len(seen)] = model.seen
        return result

    model.run_simulation = recording_run
    out = model.run_multi_scenario(config)
    return out, list(seen.values())


def test_scenario_override_gated_until_intervention():
    out, seen = run({
        'simulation.intervention_start_time': 5,
        'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
    })
    assert out['success']
    assert seen[0]['m.rate'] == [1.0] * 5 + [3.0] * 5


def test_gating_uses_calibration_value_before_intervention():
    _, seen = run({
        'simulation.intervention_start_time': 5,
        '__calibration_params__': {'m.rate': 2.0},
        'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
    })
    assert seen[0]['m.rate'] == [2.0] * 5 + [3.0] * 5


def test_calibration_applies_to_non_overridden_params_in_scenarios():
    _, seen = run({
        'simulation.intervention_start_time': 5,
        '__calibration_params__': {'m.other': 7.0},
        'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
    })
    assert seen[0]['m.other'] == [7.0] * 10


def test_calibration_applies_without_intervention_time():
    _, seen = run({
        '__calibration_params__': {'m.other': 7.0},
        'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
    })
    # No intervention time: override applies for the whole run, calibration too.
    assert seen[0]['m.rate'] == [3.0] * 10
    assert seen[0]['m.other'] == [7.0] * 10


def test_calibration_accepts_wrapped_values():
    _, seen = run({
        'simulation.intervention_start_time': 5,
        '__calibration_params__': {'m.rate': {'value': 2.0}, 'm.other': {'value': 7.0}},
        'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
    })
    assert seen[0]['m.rate'] == [2.0] * 5 + [3.0] * 5
    assert seen[0]['m.other'] == [7.0] * 10


def test_baseline_and_scenario_match_before_intervention():
    out, _ = run({
        'include_baseline': True,
        'simulation.intervention_start_time': 5,
        '__calibration_params__': {'m.rate': 2.0, 'm.other': 0.5},
        'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
    })
    base = out['scenarios']['baseline']['stocks']['m.x']
    scen = out['scenarios']['s1']['stocks']['m.x']
    # Identical through t=5 (state at t=5 only reflects pre-intervention steps).
    assert base[:6] == scen[:6]
    assert scen[-1] > base[-1]


def test_implicit_current_scenario_is_gated_and_calibrated():
    _, seen = run({
        'simulation.intervention_start_time': 5,
        '__calibration_params__': {'m.other': 7.0},
        'm.rate': 3.0,
    })
    assert seen[0]['m.rate'] == [1.0] * 5 + [3.0] * 5
    assert seen[0]['m.other'] == [7.0] * 10


def test_scenarios_do_not_leak_gating_into_each_other_or_after_run():
    model = StepModel()
    model.run_multi_scenario({
        'simulation.intervention_start_time': 5,
        'scenarios': [
            {'id': 's1', 'params': {'m.rate': 3.0}},
            {'id': 's2', 'params': {'m.other': 4.0}},
        ],
    })
    # s2 (last run) overrode only m.other, so m.rate must be ungated default.
    assert model.seen['m.rate'] == [1.0] * 10
    assert model.seen['m.other'] == [0.0] * 5 + [4.0] * 5

    # A plain run after the batch has no gating at all.
    model.run_simulation({'m.rate': 3.0, 'simulation.intervention_start_time': 5})
    assert model.seen['m.rate'] == [3.0] * 10


def test_gating_context_cleared_when_run_raises():
    model = StepModel()

    def boom(params=None):
        raise RuntimeError('fail')

    model.run_simulation = boom
    with pytest.raises(RuntimeError):
        model.run_multi_scenario({
            'simulation.intervention_start_time': 5,
            'scenarios': [{'id': 's1', 'params': {'m.rate': 3.0}}],
        })
    ctx = SimulationContext(0.0, datetime(2024, 1, 1), {'m.rate': 3.0}, {})
    assert ctx.intervention_start_time is None
    assert ctx.get_param('m.rate') == 3.0


def test_explicit_context_kwargs_take_precedence():
    ctx = SimulationContext(
        current_time=1.0,
        current_datetime=datetime(2024, 1, 1),
        all_params={'m.rate': 3.0},
        all_stocks={},
        intervention_start_time=2.0,
        scenario_param_keys={'m.rate'},
        param_defaults={'m.rate': 1.0},
    )
    assert ctx.get_param('m.rate') == 1.0
