import { prisma } from "../db/prisma";

export async function resolvePatientIdForUser(
  role: "ADMIN" | "CLINICIAN" | "PATIENT" | "SERVICE",
  userId: string,
  requestedPatientId?: string | null
): Promise<string> {
  if (role === "PATIENT") {
    const patient = await prisma.patient.findUnique({ where: { userId } });
    if (!patient) throw new Error("No patient profile found for current user");
    if (requestedPatientId && requestedPatientId !== patient.id) {
      throw new Error("Patients can only access their own data");
    }
    return patient.id;
  }

  if (!requestedPatientId) {
    throw new Error("patientId required");
  }
  return requestedPatientId;
}
