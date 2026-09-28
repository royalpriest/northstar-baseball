import assert from 'node:assert/strict';
import test from 'node:test';
import { birthplace, canadaSummary, lensLeaders } from '../lib/canada.ts';

test('RE24 coverage includes zero and negative values, excluding missing / invalid values', () => {
  assert.deepEqual(canadaSummary([
    { re24_per_100_pa: 0, season_plate_appearances: 400 },
    { re24_per_100_pa: -2, season_plate_appearances: 200 },
    { re24_per_100_pa: null }, { re24_per_100_pa: NaN }, {},
  ]), { availableRe24: 2, largestSample: 400 });
});
test('empty cohorts have no largest sample or invented leader', () => {
  assert.deepEqual(canadaSummary([]), { availableRe24: 0, largestSample: null });
  assert.deepEqual(lensLeaders([{ selection_context_percentile: null }], 'selection_context_percentile'), []);
});
test('each lens keeps its own leaders and exact ties without treating missing values as zero', () => {
  const players = [
    { batter: 1, selection_context_percentile: 90, contact_context_percentile: 10 },
    { batter: 2, selection_context_percentile: 90, contact_context_percentile: 95 },
    { batter: 3, damage_context_percentile: 0 },
  ];
  assert.deepEqual(lensLeaders(players, 'selection_context_percentile').map(p => p.batter), [1, 2]);
  assert.deepEqual(lensLeaders(players, 'contact_context_percentile').map(p => p.batter), [2]);
  assert.deepEqual(lensLeaders(players, 'damage_context_percentile').map(p => p.batter), [3]);
});
test('birthplace expands provinces and handles incomplete metadata', () => {
  assert.equal(birthplace({ birth_city: 'Montreal', birth_state_province: 'QC' }), 'Montreal, Quebec');
  assert.equal(birthplace({ birth_state_province: ' bc ' }), 'British Columbia');
  assert.equal(birthplace({ birth_city: 'Toronto' }), 'Toronto');
  assert.equal(birthplace({ birth_state_province: 'Unknown region' }), 'Unknown region');
  assert.equal(birthplace({}), 'Birthplace detail unavailable');
});
