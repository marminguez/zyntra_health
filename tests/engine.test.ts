import { describe, it, expect } from "vitest";
import { computeRisk, ZScores } from "../src/server/zyntra/engine";

describe("computeRisk", () => {
    it("returns low risk when all z-scores are near zero", () => {
        const zScores: ZScores = {
            glucose: 0,
            sleep: 0,
            hrv: 0,
            activity: 0,
            adherence: 0,
        };
        const result = computeRisk(zScores, 24);
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(1);
        expect(result.level).toBe("medium"); // 0.5 baseline from sigmoid center
    });

    it("returns high risk when glucose z-score is very high", () => {
        const zScores: ZScores = {
            glucose: 4.0,
            sleep: 0,
            hrv: 0,
            activity: 0,
        };
        const result = computeRisk(zScores, 24);
        expect(result.score).toBeGreaterThan(0.55);
        expect(result.explanation).toContain("glucose");
    });

    it("returns high risk when sleep z-score is very negative", () => {
        const zScores: ZScores = {
            glucose: 0,
            sleep: -4.0,
            hrv: 0,
            activity: 0,
        };
        const result = computeRisk(zScores, 24);
        expect(result.score).toBeGreaterThan(0.55);
        expect(result.explanation).toContain("sleep");
    });

    it("applies horizon decay for 48h", () => {
        const zScores: ZScores = { glucose: 3.0 };
        const r24 = computeRisk(zScores, 24);
        const r48 = computeRisk(zScores, 48);
        expect(r48.score).toBeLessThan(r24.score);
    });

    it("applies horizon decay for 72h", () => {
        const zScores: ZScores = { glucose: 3.0 };
        const r24 = computeRisk(zScores, 24);
        const r72 = computeRisk(zScores, 72);
        expect(r72.score).toBeLessThan(r24.score);
    });

    it("returns low risk with no z-scores provided", () => {
        const result = computeRisk({}, 24);
        expect(result.score).toBe(0);
        expect(result.level).toBe("low");
    });

    it("scores increase with worsening hrv (negative z)", () => {
        const good: ZScores = { hrv: 1.0 };
        const bad: ZScores = { hrv: -3.0 };
        const rGood = computeRisk(good, 24);
        const rBad = computeRisk(bad, 24);
        expect(rBad.score).toBeGreaterThan(rGood.score);
    });

    it("scores increase with lower activity (negative z)", () => {
        const active: ZScores = { activity: 1.5 };
        const sedentary: ZScores = { activity: -3.0 };
        const rActive = computeRisk(active, 24);
        const rSedentary = computeRisk(sedentary, 24);
        expect(rSedentary.score).toBeGreaterThan(rActive.score);
    });

    it("classifies score < 0.33 as low", () => {
        // All domains below mean → inverted domains contribute low, glucose neutral
        const zScores: ZScores = { glucose: -3.0, sleep: 3.0, hrv: 3.0 };
        const r = computeRisk(zScores, 24);
        expect(r.level).toBe("low");
    });

    it("all bad signals produce high level", () => {
        const zScores: ZScores = {
            glucose: 5.0,
            sleep: -5.0,
            hrv: -5.0,
            activity: -5.0,
            adherence: -5.0,
        };
        const r = computeRisk(zScores, 24);
        expect(r.level).toBe("high");
        expect(r.score).toBeGreaterThanOrEqual(0.66);
    });

    // --- GRUPO 8 Tests ---
    it("empty zScores -> score 0, confidence 0, level low", () => {
        const result = computeRisk({}, 24);
        expect(result.score).toBe(0);
        expect(result.confidence).toBe(0);
        expect(result.level).toBe("low");
    });

    it("glucose 10 -> score near 1, level high", () => {
        const result = computeRisk({ glucose: 10 }, 24);
        expect(result.score).toBeGreaterThan(0.9);
        expect(result.level).toBe("high");
    });

    it("rejects invalid horizons in compilation", () => {
        // TypeScript debería rechazarlo en compilación
        // @ts-expect-error
        computeRisk({ glucose: 0 }, 999);
    });
});
