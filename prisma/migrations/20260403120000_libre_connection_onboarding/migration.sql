-- CreateTable
CREATE TABLE "LibreConnection" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "invitedEmail" TEXT NOT NULL,
    "acceptedEmail" TEXT,
    "status" TEXT NOT NULL DEFAULT 'NOT_STARTED',
    "inviteSentAt" DATETIME,
    "acceptedAt" DATETIME,
    "firstDataAt" DATETIME,
    "lastDataAt" DATETIME,
    "lastCheckAt" DATETIME,
    "errorCode" TEXT,
    "errorMessage" TEXT,
    "diagnostics" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "LibreConnection_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "LibreConnection_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "Patient" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "LibreConnection_patientId_key" ON "LibreConnection"("patientId");

-- CreateIndex
CREATE INDEX "LibreConnection_userId_idx" ON "LibreConnection"("userId");
