/**
 * Simple in-memory rate limiter keyed by IP + route.
 * Uses a sliding window counter reset every minute.
 */

interface RateBucket {
    count: number;
    resetAt: number;
}

const store = new Map<string, RateBucket>();

const WINDOW_MS = 60_000; // 1 minute

function getLimit(route: string): number {
    if (route === "ingest") return Number(process.env.RATE_LIMIT_INGEST) || 60;
    if (route === "risk") return Number(process.env.RATE_LIMIT_RISK) || 60;
    return 60;
}

/**
 * Returns `true` if the request is allowed, `false` if rate-limited.
 */
export function checkRateLimit(ip: string, route: string): boolean {
    // NOTA: este rate limiter es in-memory y no es consistente entre múltiples
    // instancias (Vercel, Railway con >1 replica). Para producción multi-instancia,
    // sustituir por Redis con ioredis + sliding window.
    // TODO: RATE_LIMITER_BACKEND=redis
    
    const key = `${ip}:${route}`;
    const now = Date.now();
    const limit = getLimit(route);

    let bucket = store.get(key);

    if (!bucket || now > bucket.resetAt) {
        bucket = { count: 0, resetAt: now + WINDOW_MS };
        store.set(key, bucket);
    }

    bucket.count += 1;
    return bucket.count <= limit;
}

/** Periodic cleanup (call from a setInterval if desired). */
export function pruneExpired(): void {
    const now = Date.now();
    for (const [key, bucket] of store) {
        if (now > bucket.resetAt) store.delete(key);
    }
}

// Auto-cleanup cada 5 minutos. .unref() permite que Node.js salga limpiamente.
if (typeof setInterval !== "undefined") {
  const _pruneTimer = setInterval(pruneExpired, 5 * 60_000);
  _pruneTimer.unref?.();
}
