import { execSync } from "child_process";
import path from "path";

export const HYPOGLYCEMIA_LSTM_LOOKBACK = 48;
export const HYPOGLYCEMIA_LSTM_FEATURES = ["glucose", "bolus", "carbs_g", "step_count", "iob"] as const;

export type HypoglycemiaLstmFeature = typeof HYPOGLYCEMIA_LSTM_FEATURES[number];

export type HypoglycemiaLstmStep = Record<HypoglycemiaLstmFeature, number>;

export interface HypoglycemiaLstmResult {
  probability: number;
  threshold: number;
  alert: boolean;
  lookback: number;
  features: string[];
  modelVersion: string;
  scalerUsed: boolean;
  fallback: boolean;
  error?: string;
}

interface PythonHypoglycemiaLstmResult {
  probability?: number;
  threshold?: number;
  alert?: boolean;
  lookback?: number;
  features?: string[];
  model_version?: string;
  scaler_used?: boolean;
  fallback?: boolean;
  error?: string;
}

function defaultThreshold(): number {
  const configured = Number(process.env.HYPO_LSTM_THRESHOLD);
  return Number.isFinite(configured) ? configured : 0.5;
}

function fallbackResult(error: string): HypoglycemiaLstmResult {
  return {
    probability: 0,
    threshold: defaultThreshold(),
    alert: false,
    lookback: HYPOGLYCEMIA_LSTM_LOOKBACK,
    features: [...HYPOGLYCEMIA_LSTM_FEATURES],
    modelVersion: "lstm_hypoglycemia_unavailable",
    scalerUsed: false,
    fallback: true,
    error,
  };
}

export function hasEnoughHypoglycemiaLstmSteps(sequence: HypoglycemiaLstmStep[]): boolean {
  return sequence.length >= HYPOGLYCEMIA_LSTM_LOOKBACK;
}

export function predictHypoglycemiaLstm(sequence: HypoglycemiaLstmStep[]): HypoglycemiaLstmResult {
  if (!hasEnoughHypoglycemiaLstmSteps(sequence)) {
    return fallbackResult(`At least ${HYPOGLYCEMIA_LSTM_LOOKBACK} timesteps are required`);
  }

  const input = JSON.stringify({
    sequence: sequence.slice(-HYPOGLYCEMIA_LSTM_LOOKBACK),
    threshold: defaultThreshold(),
  });

  try {
    const scriptPath = path.resolve("ml/predict_lstm_hypoglycemia.py");
    const pythonCmd = process.env.PYTHON_BIN ?? (process.platform === "win32" ? "python" : "python3");
    const output = execSync(`${pythonCmd} "${scriptPath}"`, {
      input,
      encoding: "utf-8",
      timeout: 10_000,
      env: { ...process.env },
    });

    const parsed = JSON.parse(output.trim()) as PythonHypoglycemiaLstmResult;
    if (parsed.fallback || parsed.error || typeof parsed.probability !== "number") {
      return fallbackResult(parsed.error ?? "LSTM hypoglycemia inference returned no probability");
    }

    const threshold = typeof parsed.threshold === "number" ? parsed.threshold : defaultThreshold();
    const probability = Math.max(0, Math.min(1, parsed.probability));

    return {
      probability,
      threshold,
      alert: typeof parsed.alert === "boolean" ? parsed.alert : probability >= threshold,
      lookback: parsed.lookback ?? HYPOGLYCEMIA_LSTM_LOOKBACK,
      features: parsed.features ?? [...HYPOGLYCEMIA_LSTM_FEATURES],
      modelVersion: parsed.model_version ?? "lstm_hypoglycemia_classifier_v1",
      scalerUsed: parsed.scaler_used ?? false,
      fallback: false,
    };
  } catch (error) {
    return fallbackResult(error instanceof Error ? error.message : String(error));
  }
}
