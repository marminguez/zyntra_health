import fs from 'fs';
import path from 'path';

export interface MLModel {
    feature_names: string[];
    scaler_mean: number[];
    scaler_scale: number[];
    coef: number[];
    intercept: number;
}

export interface Driver {
    feature: string;
    contribution: number;
    direction: "up" | "down";
}

export interface PredictionResult {
    probability: number;
    top_drivers: Driver[];
}

const TARGETS = [
    "y_hyper_24",
    "y_hyper_72",
    "y_hypo_12",
    "y_hypo_24",
    "y_tir_drop_24",
    "y_tir_drop_72",
    "y_instability_24",
    "y_instability_72"
] as const;

export type Target = typeof TARGETS[number];

const modelCache = new Map<Target, MLModel>();

export function loadModels(): boolean {
    try {
        const modelsDir = path.join(process.cwd(), "ml", "models");
        for (const target of TARGETS) {
            const modelPath = path.join(modelsDir, `${target}.json`);
            if (!fs.existsSync(modelPath)) return false;

            const data = fs.readFileSync(modelPath, "utf-8");
            const model = JSON.parse(data) as MLModel;
            modelCache.set(target, model);
        }
        return true;
    } catch (err) {
        console.error("Error loading ML models:", err);
        return false;
    }
}

export function clamp(val: number, min: number, max: number): number {
    return Math.max(min, Math.min(val, max));
}

export function computePCI(
    delta_hrv: number,
    delta_sleep: number,
    delta_stress: number,
    delta_resting_hr: number,
    delta_temp: number
): number {
    const c1 = clamp(-delta_hrv, -1.0, 1.0);
    const c2 = clamp(-delta_sleep, -1.0, 1.0);
    const c3 = clamp(delta_stress, -1.0, 1.0);
    const c4 = clamp(delta_resting_hr, -1.0, 1.0);
    const c5 = clamp(delta_temp, -1.0, 1.0);

    const pci_raw = 0.3 * c1 + 0.2 * c2 + 0.2 * c3 + 0.15 * c4 + 0.15 * c5;
    return 1.0 / (1.0 + Math.exp(-pci_raw));
}

export function predictTarget(target: Target, features: Record<string, number>): PredictionResult | null {
    const model = modelCache.get(target);
    if (!model) return null;

    let logit = model.intercept;
    const drivers: Driver[] = [];

    for (let i = 0; i < model.feature_names.length; i++) {
        const fName = model.feature_names[i];
        const val = features[fName] ?? 0;

        const mean = model.scaler_mean[i];
        const scale = model.scaler_scale[i];
        const coef = model.coef[i];

        // handle 0 scale
        const scaled = scale === 0 ? 0 : (val - mean) / scale;
        const contribution = coef * scaled;

        logit += contribution;

        drivers.push({
            feature: fName,
            contribution: Math.abs(contribution),
            direction: contribution > 0 ? "up" : "down"
        });
    }

    // sort drivers by absolute contribution descending
    drivers.sort((a, b) => b.contribution - a.contribution);

    const probability = 1.0 / (1.0 + Math.exp(-logit));

    return {
        probability,
        top_drivers: drivers.slice(0, 3) // keep top 3
    };
}
