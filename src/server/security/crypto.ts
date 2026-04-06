import crypto from "crypto";

const SENSITIVE_TYPES = new Set([
    "cgm_glucose_mgdl",
    "hr_bpm",
    "hrv_rmssd",
    "sleep_minutes",
    "steps",
]);

function getKey(): Buffer {
    const b64 = process.env.ZYNTRA_FIELD_ENCRYPTION_KEY;
    if (!b64) throw new Error("ZYNTRA_FIELD_ENCRYPTION_KEY not set");
    const key = Buffer.from(b64, "base64");
    if (key.length !== 32) {
        throw new Error(`Encryption key must be 32 bytes (got ${key.length})`);
    }
    return key;
}

export function isSensitiveType(type: string): boolean {
    return SENSITIVE_TYPES.has(type);
}

export async function encryptValue(value: string | number): Promise<string> {
    const key = getKey();
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
    
    let encrypted = cipher.update(String(value), "utf8");
    encrypted = Buffer.concat([encrypted, cipher.final()]);
    const authTag = cipher.getAuthTag();
    
    const combined = Buffer.concat([iv, authTag, encrypted]);
    return combined.toString("base64");
}

export async function decryptValue(encoded: string): Promise<string> {
    const key = getKey();
    const combined = Buffer.from(encoded, "base64");
    
    const iv = combined.subarray(0, 12);
    const authTag = combined.subarray(12, 28);
    const encrypted = combined.subarray(28);
    
    const decipher = crypto.createDecipheriv("aes-256-gcm", key, iv);
    decipher.setAuthTag(authTag);
    
    let decrypted = decipher.update(encrypted);
    decrypted = Buffer.concat([decrypted, decipher.final()]);
    
    return decrypted.toString("utf8");
}

export async function decryptNumber(encoded: string): Promise<number> {
    const value = await decryptValue(encoded);
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        throw new Error("Decrypted value is not a valid number");
    }
    return parsed;
}
