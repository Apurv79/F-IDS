# F-IDS Backend Service

Backend microservice for the Federated Intrusion Detection System (F-IDS) built with **Node.js**, **Express**, **TypeScript**, **MongoDB**, **Redis**, and **BullMQ**.

## Features
- **Native ES Modules**: Strict TypeScript configuration (`NodeNext` / `ES2022`).
- **Modular Architecture**: Feature-sliced structure (`auth`, `node`, `attack`, `model`, `alert`).
- **Real-Time Communication**: Socket.IO server for streaming alerts and node telemetry.
- **Background Processing**: BullMQ queues for alert notifications and model training jobs.
- **Microservice Integration**: Axios interface for Python ML inference service.

## Getting Started

### Installation
```bash
cd backend
npm install
```

### Environment Variables
Copy `.env.example` to `.env` and adjust settings:
```bash
cp .env.example .env
```

### Development Server
```bash
npm run dev
```

### Background Worker Process
```bash
npm run worker
```

### Production Build
```bash
npm run build
npm start
```
