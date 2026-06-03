import { describe, expect, it } from "vitest";
import {
  hasEnoughHypoglycemiaLstmSteps,
  HYPOGLYCEMIA_LSTM_LOOKBACK,
  predictHypoglycemiaLstm,
  type HypoglycemiaLstmStep,
} from "../src/server/zyntra/hypoglycemiaLstm";

function makeStep(glucose: number, bolus = 0): HypoglycemiaLstmStep {
  return {
    glucose,
    bolus,
    carbs_g: 0,
    step_count: 0,
    iob: bolus,
  };
}

describe("hypoglycemia LSTM wrapper", () => {
  it("requires the 48-step LSTM lookback window", () => {
    const shortSequence = Array.from({ length: HYPOGLYCEMIA_LSTM_LOOKBACK - 1 }, () => makeStep(110));
    const fullSequence = Array.from({ length: HYPOGLYCEMIA_LSTM_LOOKBACK }, () => makeStep(110));

    expect(hasEnoughHypoglycemiaLstmSteps(shortSequence)).toBe(false);
    expect(hasEnoughHypoglycemiaLstmSteps(fullSequence)).toBe(true);
  });

  it("returns a safe fallback instead of throwing when the sequence is too short", () => {
    const result = predictHypoglycemiaLstm([makeStep(110)]);

    expect(result.fallback).toBe(true);
    expect(result.alert).toBe(false);
    expect(result.probability).toBe(0);
    expect(result.error).toContain("48 timesteps");
  });
});
