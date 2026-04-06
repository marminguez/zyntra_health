import fs from 'fs';
import path from 'path';

export interface GlucoseRecord {
  timestamp: Date;
  value: number;
}

export interface ActivityRecord {
  timestamp: Date;
  steps: number;
}

export interface SleepRecord {
  startTime: Date;
  endTime: Date;
  durationMs: number;
}

export interface InsulinRecord {
  timestamp: Date;
  type: 'basal' | 'bolus';
  units: number;
}

export interface NutritionRecord {
  timestamp: Date;
  carbs: number;
  protein: number;
  fat: number;
}

export interface ParticipantData {
  participantId: string;
  glucose: GlucoseRecord[];
  activity: ActivityRecord[];
  sleep: SleepRecord[];
  insulin: InsulinRecord[];
  nutrition: NutritionRecord[];
}

export class DataAgent {
  private static instance: DataAgent;
  private cache: Map<string, ParticipantData> = new Map();
  private dataRoot = 'd:/HARVARD/zyntrahack/data/raw/demo/';

  private constructor() {}

  public static getInstance(): DataAgent {
    if (!DataAgent.instance) {
      DataAgent.instance = new DataAgent();
    }
    return DataAgent.instance;
  }

  public async getParticipantData(participantId: string): Promise<ParticipantData | null> {
    if (participantId !== '2302') {
      console.warn(`[DataAgent] Demo only supports participant 2302. Requested: ${participantId}`);
      return null;
    }

    if (this.cache.has(participantId)) {
      console.log(`[DataAgent] Returning cached data for participant ${participantId}`);
      return this.cache.get(participantId)!;
    }

    console.log(`[DataAgent] Loading data for participant ${participantId}...`);
    const data = await this.loadFromCSV(participantId);
    if (data) {
      this.cache.set(participantId, data);
    }
    return data;
  }

  private async loadFromCSV(participantId: string): Promise<ParticipantData | null> {
    const files = {
      glucose: `UoMGlucose${participantId}.csv`,
      activity: `UoMActivity${participantId}.csv`,
      sleep: `UoM2302sleeptime.csv`, // Note specific filename for sleep
      basal: `UoMBasal${participantId}.csv`,
      bolus: `UoMBolus${participantId}.csv`,
      nutrition: `UoMNutrition${participantId}.csv`
    };

    const data: ParticipantData = {
      participantId,
      glucose: [],
      activity: [],
      sleep: [],
      insulin: [],
      nutrition: []
    };

    try {
      // 1. Glucose
      data.glucose = this.parseCSV<GlucoseRecord>(path.join(this.dataRoot, files.glucose), (cols) => ({
        timestamp: new Date(cols[0]),
        value: parseFloat(parseFloat(cols[1]).toFixed(2)) * 18 // Unit conversion: mmol/L to mg/dL
      }));

      // 2. Activity
      data.activity = this.parseCSV<ActivityRecord>(path.join(this.dataRoot, files.activity), (cols) => ({
        timestamp: new Date(cols[0]),
        steps: parseInt(cols[1]) || 0
      }));

      // 3. Sleep
      data.sleep = this.parseCSV<SleepRecord>(path.join(this.dataRoot, files.sleep), (cols) => ({
        startTime: new Date(cols[0]),
        endTime: new Date(cols[1]),
        durationMs: parseInt(cols[2]) || 0
      }));

      // 4. Basal
      const basal = this.parseCSV<InsulinRecord>(path.join(this.dataRoot, files.basal), (cols) => ({
        timestamp: new Date(cols[0]),
        type: 'basal',
        units: parseFloat(cols[1]) || 0
      }));

      // 5. Bolus
      const bolus = this.parseCSV<InsulinRecord>(path.join(this.dataRoot, files.bolus), (cols) => ({
        timestamp: new Date(cols[0]),
        type: 'bolus',
        units: parseFloat(cols[1]) || 0
      }));
      data.insulin = [...basal, ...bolus];

      // 6. Nutrition
      data.nutrition = this.parseCSV<NutritionRecord>(path.join(this.dataRoot, files.nutrition), (cols) => ({
        timestamp: new Date(cols[0]),
        carbs: parseFloat(cols[1]) || 0,
        protein: parseFloat(cols[2]) || 0,
        fat: parseFloat(cols[3]) || 0
      }));

      console.log(`[DataAgent] Successfully loaded data for ${participantId}:`);
      console.log(`- Glucose: ${data.glucose.length} rows`);
      console.log(`- Activity: ${data.activity.length} rows`);
      console.log(`- Sleep: ${data.sleep.length} rows`);
      console.log(`- Insulin: ${data.insulin.length} rows`);
      console.log(`- Nutrition: ${data.nutrition.length} rows`);

      return data;
    } catch (error) {
      console.error(`[DataAgent] Error loading CSV files for ${participantId}:`, error);
      return null;
    }
  }

  private parseCSV<T>(filePath: string, mapper: (columns: string[]) => T): T[] {
    if (!fs.existsSync(filePath)) {
      console.warn(`[DataAgent] File not found: ${filePath}`);
      return [];
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');
    const records: T[] = [];

    // Skip header (assuming first line is header)
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const columns = line.split(',');
        try {
            const record = mapper(columns);
            // Basic validation
            if (record) records.push(record);
        } catch (e) {
            // Silently skip malformed rows for now
        }
    }

    return records;
  }
}
