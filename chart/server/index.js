/* eslint-env node */
import 'dotenv/config';
import express from 'express';
import { MongoClient } from 'mongodb';

const app = express();
const PORT = 3001;
const MONGODB_URI = process.env.MONGO; // read from env instead of hardcoded string
const DB_NAME = 'FinamHackathon';
const COLLECTION = 'news';

let mongoClient;
let cachedDb;

async function getDb() {
  if (!MONGODB_URI) return null; // no DB in local dev without env
  if (cachedDb) return cachedDb;
  mongoClient = new MongoClient(MONGODB_URI);
  await mongoClient.connect();
  cachedDb = mongoClient.db(DB_NAME);
  return cachedDb;
}

function toHierarchy(rows) {
  return {
    name: 'root',
    children: rows.map(r => ({
      name: r.name,
      value: Math.max(0, Number(r.value) || 0),
      sentiment: Math.max(-1, Math.min(1, Number(r.sentiment)) || 0),
      count: r.count || 0,
    }))
  };
}

// Fallback mock in case there is no DB configured
function mockData() {
  const sample = [
    { name: 'AAPL', value: 12.5, sentiment: 0.2, count: 20 },
    { name: 'MSFT', value: 9.1, sentiment: 0.6, count: 15 },
    { name: 'GOOGL', value: 7.7, sentiment: -0.1, count: 12 },
    { name: 'AMZN', value: 14.4, sentiment: 0.0, count: 22 },
    { name: 'TSLA', value: 5.2, sentiment: -0.4, count: 8 },
    { name: 'NVDA', value: 11.3, sentiment: 0.8, count: 10 }
  ];
  return toHierarchy(sample);
}

app.get('/api/tickers', async (req, res) => {
  try {
    const db = await getDb();
    if (!db) {
      return res.json(mockData());
    }

    const pipeline = [
      { $unwind: { path: '$entities.tickers', preserveNullAndEmptyArrays: false } },
      {
        $group: {
          _id: '$entities.tickers',
          totalHottness: { $sum: { $ifNull: ['$hottness.score', 0] } },
          avgSentiment: { $avg: { $ifNull: ['$sentiment_score', 0] } },
          count: { $sum: 1 }
        }
      },
      {
        $project: {
          _id: 0,
          name: '$_id',
          value: { $ifNull: ['$totalHottness', 0] },
          sentiment: { $ifNull: ['$avgSentiment', 0] },
          count: 1
        }
      },
      { $sort: { value: -1 } },
      { $limit: 500 }
    ];

    const rows = await db.collection(COLLECTION).aggregate(pipeline).toArray();
    return res.json(toHierarchy(rows));
  } catch (err) {
    console.error('Aggregation error:', err);
    return res.status(500).json({ error: 'Failed to aggregate data' });
  }
});

app.get('/api/companies', async (req, res) => {
  try {
    const db = await getDb();
    if (!db) {
      return res.json(mockData());
    }

    const pipeline = [
      { $unwind: { path: '$entities.companies', preserveNullAndEmptyArrays: false } },
      {
        $group: {
          _id: '$entities.companies',
          totalHottness: { $sum: { $ifNull: ['$hottness.score', 0] } },
          avgSentiment: { $avg: { $ifNull: ['$sentiment_score', 0] } },
          count: { $sum: 1 }
        }
      },
      {
        $project: {
          _id: 0,
          name: '$_id',
          value: { $ifNull: ['$totalHottness', 0] },
          sentiment: { $ifNull: ['$avgSentiment', 0] },
          count: 1
        }
      },
      { $sort: { value: -1 } },
      { $limit: 500 }
    ];

    const rows = await db.collection(COLLECTION).aggregate(pipeline).toArray();
    return res.json(toHierarchy(rows));
  } catch (err) {
    console.error('Aggregation error:', err);
    return res.status(500).json({ error: 'Failed to aggregate data' });
  }
});

app.get('/health', (req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`API server listening on http://localhost:${PORT}`);
});
