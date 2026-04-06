/**
 * Seed script – creates demo users, patients, and consent records.
 *
 * Run: npx ts-node --compiler-options '{"module":"CommonJS"}' scripts/seed.ts
 */

import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
    console.log("🌱 Seeding Zyntra database…");

    // ── Users ──────────────────────────────────────────────────
    const admin = await prisma.user.upsert({
        where: { email: "admin@zyntra.dev" },
        update: {},
        create: {
            email: "admin@zyntra.dev",
            name: "Admin",
            role: "ADMIN",
        },
    });

    const clinician = await prisma.user.upsert({
        where: { email: "clinician@zyntra.dev" },
        update: {},
        create: {
            email: "clinician@zyntra.dev",
            name: "Dr. Smith",
            role: "CLINICIAN",
        },
    });

    const patientUser = await prisma.user.upsert({
        where: { email: "patient@zyntra.dev" },
        update: {},
        create: {
            email: "patient@zyntra.dev",
            name: "Jane Doe",
            role: "PATIENT",
        },
    });

    console.log(`  Users: ${admin.id}, ${clinician.id}, ${patientUser.id}`);

    // ── Patient profile ────────────────────────────────────────
    const patient = await prisma.patient.upsert({
        where: { userId: patientUser.id },
        update: {},
        create: {
            userId: patientUser.id,
        },
    });

    console.log(`  Patient: ${patient.id}`);

    // ── Consent ────────────────────────────────────────────────
    const consent = await prisma.consent.create({
        data: {
            patientId: patient.id,
            allowCGM: true,
            allowWearable: true,
            allowManual: true,
            policyVersion: "1.0",
        },
    });

    console.log(`  Consent: ${consent.id}`);

    // ── Sample signals (non-sensitive so value is stored in plain) ──
    const now = new Date();
    const signalTypes = [
        { type: "body_temp_c", source: "WEARABLE" as const, value: 36.8, unit: "°C" },
        { type: "spo2_pct", source: "WEARABLE" as const, value: 97.5, unit: "%" },
        { type: "weight_kg", source: "MANUAL" as const, value: 72.1, unit: "kg" },
    ];

    for (const s of signalTypes) {
        await prisma.signal.create({
            data: {
                patientId: patient.id,
                source: s.source,
                type: s.type,
                ts: now,
                value: s.value,
                unit: s.unit,
            },
        });
    }

    console.log(`  Created ${signalTypes.length} sample signals`);
    console.log("✅ Seed complete");
}

main()
    .catch((e) => {
        console.error(e);
        process.exit(1);
    })
    .finally(() => prisma.$disconnect());
