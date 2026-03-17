# 🚀 MERN Stack — Complete Full-Stack Development Curriculum

<div align="center">

![MERN Stack](https://img.shields.io/badge/MERN-Stack-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Express.js](https://img.shields.io/badge/Express.js-000000?style=for-the-badge&logo=express&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)

![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![NestJS](https://img.shields.io/badge/NestJS-E0234E?style=for-the-badge&logo=nestjs&logoColor=white)
![NGINX](https://img.shields.io/badge/NGINX-009639?style=for-the-badge&logo=nginx&logoColor=white)

**A comprehensive, industry-grade learning repository covering every layer of modern full-stack JavaScript development — from raw HTTP fundamentals to production-ready deployment.**

[📚 Get Started](#-getting-started) · [🗺️ Learning Path](#%EF%B8%8F-recommended-learning-path) · [📂 Modules](#-repository-structure) · [🤝 Contributing](#-contributing)

</div>

---

## 📖 Table of Contents

1. [What Is This Repository?](#-what-is-this-repository)
2. [What Is the MERN Stack?](#-what-is-the-mern-stack)
3. [Prerequisites](#-prerequisites)
4. [Repository Structure](#-repository-structure)
5. [Recommended Learning Path](#%EF%B8%8F-recommended-learning-path)
6. [Module Deep Dives](#-module-deep-dives)
   - [JavaScript Fundamentals](#1-javascript-fundamentals)
   - [HTTP & HTTPS Fundamentals](#2-http--https-fundamentals)
   - [Node.js Fundamentals](#3-nodejs-fundamentals)
   - [Express Fundamentals](#4-express-fundamentals)
   - [NoSQL Fundamentals (MongoDB)](#5-nosql-fundamentals-mongodb)
   - [SQL Fundamentals](#6-sql-fundamentals)
   - [React Fundamentals](#7-react-fundamentals)
   - [Next.js Fundamentals](#8-nextjs-fundamentals)
   - [NestJS Fundamentals](#9-nestjs-fundamentals)
   - [React Native Fundamentals](#10-react-native-fundamentals)
   - [Vue.js Fundamentals](#11-vuejs-fundamentals)
   - [Bun.js Fundamentals](#12-bunjs-fundamentals)
   - [NGINX Fundamentals](#13-nginx-fundamentals)
   - [Full MERN Stack Project](#14-full-mern-stack-project)
7. [Environment Setup](#-environment-setup)
8. [Core Concepts Explained](#-core-concepts-explained)
9. [Best Practices & Patterns](#-best-practices--patterns)
10. [Deployment Guide](#-deployment-guide)
11. [Troubleshooting](#-troubleshooting)
12. [Contributing](#-contributing)
13. [License](#-license)

---

## 🎯 What Is This Repository?

This repository is a **structured, end-to-end curriculum** for developers who want to master full-stack JavaScript development using the MERN stack and its surrounding ecosystem. Whether you're a complete beginner writing your first `console.log`, an intermediate developer looking to fill in the gaps, or an experienced engineer who wants a single reference for the entire stack — this repo has you covered.

Each module is self-contained and progressively builds on the previous one. The curriculum goes beyond just MERN and introduces modern production technologies such as **Next.js**, **NestJS**, **NGINX**, **Bun.js**, and **Vue.js**, giving you a holistic picture of the JavaScript ecosystem.

### ✅ What You Will Learn

- How the internet works — HTTP, HTTPS, DNS, TCP/IP, REST
- JavaScript from fundamentals to advanced (closures, async/await, prototypes, modules)
- Building servers with **Node.js** and **Express.js**
- Working with both **NoSQL (MongoDB)** and **SQL** databases
- Building reactive UIs with **React** and state management patterns
- Server-side rendering and static site generation with **Next.js**
- Enterprise-grade backend architecture with **NestJS**
- Cross-platform mobile development with **React Native**
- Reverse proxying, load balancing, and serving static assets with **NGINX**
- Rapid prototyping with **Bun.js**
- Building reactive frontends with **Vue.js**
- Connecting it all into a **full-stack MERN application**

---

## 🧩 What Is the MERN Stack?

The **MERN** stack is a set of JavaScript technologies used to build modern full-stack web applications:

```
┌─────────────────────────────────────────────────────────────────┐
│                        MERN STACK                               │
│                                                                 │
│  ┌─────────┐    ┌─────────────┐    ┌────────┐    ┌──────────┐   │
│  │  React  │───▶│  Express.js │───▶│  Node  │───▶│ MongoDB│   │
│  │(Frontend│    │  (REST API) │    │   .js  │    │(Database)│   │
│  │   UI)   │    │             │    │(Runtime│    │          │   │
│  └─────────┘    └─────────────┘    └────────┘    └──────────┘   │
│   Browser          HTTP Layer        Server         Persistence │
└─────────────────────────────────────────────────────────────────┘
```

| Letter | Technology | Role |
|--------|-----------|------|
| **M** | MongoDB | NoSQL document database for data persistence |
| **E** | Express.js | Minimalist web framework that runs on Node.js |
| **R** | React | Declarative, component-based frontend UI library |
| **N** | Node.js | JavaScript runtime that executes server-side code |

### Why MERN?

- **Single language across the entire stack** — JavaScript on both frontend and backend
- **JSON everywhere** — Data flows natively from MongoDB → Express → React without format conversion
- **Massive ecosystem** — npm has over 2 million packages
- **Highly scalable** — Used in production by Netflix, Uber, LinkedIn, and more
- **Active community** — Abundant resources, tutorials, and job opportunities

---

## 🛠 Prerequisites

Before starting, ensure you have the following installed:

### Required Software

```bash
# Node.js (v18+ recommended)
node --version   # Should output v18.x.x or higher

# npm (comes with Node)
npm --version    # Should output 9.x.x or higher

# Git
git --version

# A code editor — VS Code is strongly recommended
code --version
```

### Recommended VS Code Extensions

| Extension | Purpose |
|-----------|---------|
| **ES7+ React/Redux/React-Native snippets** | React code snippets |
| **Prettier** | Auto-formatting |
| **ESLint** | Code linting |
| **MongoDB for VS Code** | Browse MongoDB collections |
| **REST Client** | Test API endpoints |
| **Thunder Client** | Lightweight Postman alternative |
| **GitLens** | Enhanced Git history |
| **Bracket Pair Colorizer** | Visual bracket matching |

### Knowledge Assumptions

- Basic understanding of HTML & CSS
- Familiarity with programming concepts (variables, loops, conditionals)
- No prior JavaScript, Node.js, or database experience required

---

## 📂 Repository Structure

```
MernStack/
│
├── 📁 Javascript Fundamentals/     # Core JS — the language itself
├── 📁 HTTP and HTTPS Fundamentals/ # Networking and protocol layer
├── 📁 NodeJS Fundamentals/         # Server-side JS runtime
├── 📁 Express Fundamentals/        # REST API construction
├── 📁 No SQL Fundamentals/         # MongoDB & document databases
├── 📁 SQL Fundamentals/            # Relational databases
├── 📁 React Fundamentals/          # Frontend UI with React
├── 📁 NextJS Fundamentals/         # SSR & full-stack with Next.js
├── 📁 NestJS Fundamentals/         # Enterprise backend with NestJS
├── 📁 React Native Fundamentals/   # Mobile development
├── 📁 vueJS Fundamentals/          # Alternative frontend framework
├── 📁 bunJS Fundamentals/          # Modern JavaScript runtime
├── 📁 NGINX Fundamentals/          # Web server & reverse proxy
└── 📄 MernStack                    # Capstone full-stack project
```

---

## 🗺️ Recommended Learning Path

Follow this sequence for the best learning experience. Each phase builds on the previous:

```
Phase 1: Foundation (Weeks 1–2)
  └── JavaScript Fundamentals
      └── HTTP & HTTPS Fundamentals

Phase 2: Backend (Weeks 3–4)
  └── Node.js Fundamentals
      └── Express Fundamentals
          └── NoSQL Fundamentals (MongoDB)
              └── SQL Fundamentals

Phase 3: Frontend (Weeks 5–6)
  └── React Fundamentals
      └── Next.js Fundamentals

Phase 4: Full-Stack Integration (Week 7)
  └── MERN Stack Capstone Project

Phase 5: Ecosystem & Advanced Topics (Weeks 8–10)
  └── NestJS Fundamentals
  └── React Native Fundamentals
  └── Vue.js Fundamentals
  └── Bun.js Fundamentals
  └── NGINX Fundamentals
```

---

## 🔬 Module Deep Dives

---

### 1. JavaScript Fundamentals

> **Folder:** `Javascript Fundamentals/`  
> **Language:** JavaScript (ES5 → ES2024)  
> **Estimated Time:** 1–2 weeks

JavaScript is the single language that powers every part of the MERN stack. Before touching any framework, you must be solid in the language itself.

#### Topics Covered

**Core Language**
- Variables: `var`, `let`, `const` — scoping and hoisting
- Data types: primitives vs. objects, type coercion, `typeof`
- Operators: arithmetic, comparison, logical, nullish coalescing (`??`), optional chaining (`?.`)
- Control flow: `if/else`, `switch`, `for`, `while`, `for...of`, `for...in`

**Functions**
- Function declarations vs. expressions vs. arrow functions
- Higher-order functions: `map`, `filter`, `reduce`, `forEach`
- Closures — one of the most important and misunderstood concepts in JS
- Currying and partial application
- IIFE (Immediately Invoked Function Expressions)

**Objects & Prototypes**
- Object creation patterns: literal, `Object.create()`, classes
- `this` keyword — context and binding rules
- Prototype chain and inheritance
- ES6 Classes — syntactic sugar over prototypes
- Destructuring, spread/rest operators

**Asynchronous JavaScript**
- The event loop — how JS handles concurrency
- Callbacks and callback hell
- Promises: `.then()`, `.catch()`, `.finally()`, `Promise.all()`, `Promise.race()`
- `async/await` — the modern, readable way to handle async code
- Error handling with `try/catch`

**ES6+ Features**
- Template literals
- Modules: `import` / `export` (ESM) vs. `require()` / `module.exports` (CJS)
- Symbol, Map, Set, WeakMap, WeakSet
- Generators and Iterators
- Proxy and Reflect API

#### Key Code Examples

```javascript
// ✅ Closures — a function that "remembers" its outer scope
function makeCounter() {
  let count = 0; // This variable is enclosed
  return {
    increment: () => ++count,
    decrement: () => --count,
    value: () => count,
  };
}

const counter = makeCounter();
counter.increment(); // 1
counter.increment(); // 2
counter.decrement(); // 1
console.log(counter.value()); // 1

// ✅ Async/Await with error handling
async function fetchUserData(userId) {
  try {
    const response = await fetch(`https://api.example.com/users/${userId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch user:', error.message);
    throw error; // Re-throw so callers can handle it
  }
}

// ✅ Array methods — functional programming style
const users = [
  { name: 'Alice', age: 28, role: 'admin' },
  { name: 'Bob', age: 22, role: 'user' },
  { name: 'Carol', age: 35, role: 'admin' },
];

const adminNames = users
  .filter(user => user.role === 'admin')    // Keep only admins
  .map(user => user.name)                   // Extract names
  .sort();                                   // Alphabetical order

console.log(adminNames); // ['Alice', 'Carol']
```

---

### 2. HTTP & HTTPS Fundamentals

> **Folder:** `HTTP and HTTPS Fundamentals/`  
> **Estimated Time:** 3–4 days

Understanding how data travels over the internet is essential to becoming a backend developer. Every API call, every browser request, every form submission follows these rules.

#### Topics Covered

**HTTP Protocol**
- Request-Response cycle
- HTTP Methods: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, `OPTIONS`, `HEAD`
- URL anatomy: protocol, domain, path, query string, fragment
- Request structure: start line, headers, body
- Response structure: status line, headers, body

**HTTP Status Codes**
| Range | Category | Common Examples |
|-------|----------|----------------|
| 1xx | Informational | `100 Continue` |
| 2xx | Success | `200 OK`, `201 Created`, `204 No Content` |
| 3xx | Redirection | `301 Moved Permanently`, `302 Found` |
| 4xx | Client Error | `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `422 Unprocessable Entity` |
| 5xx | Server Error | `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable` |

**HTTP Headers**
- Request headers: `Content-Type`, `Authorization`, `Accept`, `User-Agent`, `Cookie`
- Response headers: `Set-Cookie`, `Cache-Control`, `Content-Type`, `CORS headers`
- Custom headers: `X-Request-ID`, `X-API-Key`

**HTTPS & Security**
- TLS/SSL — how encryption works
- Certificates and Certificate Authorities (CAs)
- How the TLS handshake establishes a secure connection
- HTTP/2 and HTTP/3 improvements

**REST Architecture**
- Statelessness
- Resource-based URLs
- HATEOAS (Hypermedia as the Engine of Application State)
- REST vs. GraphQL vs. gRPC

#### REST API Design Principles

```
✅ Good REST URL Design:
  GET    /api/v1/users           → List all users
  POST   /api/v1/users           → Create a user
  GET    /api/v1/users/:id       → Get specific user
  PUT    /api/v1/users/:id       → Replace user entirely
  PATCH  /api/v1/users/:id       → Update part of user
  DELETE /api/v1/users/:id       → Delete a user
  GET    /api/v1/users/:id/posts → Get user's posts (nested resource)

❌ Bad REST URL Design:
  GET  /getUsers
  POST /createUser
  GET  /deleteUser?id=123
```

---

### 3. Node.js Fundamentals

> **Folder:** `NodeJS Fundamentals/`  
> **Language:** JavaScript  
> **Estimated Time:** 1 week

Node.js is a **JavaScript runtime built on Chrome's V8 engine** that lets you run JavaScript outside the browser — specifically on a server.

#### Why Node.js for Backend?

- **Non-blocking I/O** — Node doesn't wait for database queries or file reads to complete. It moves on and handles the result via callbacks/promises, making it highly efficient for I/O-heavy applications.
- **Single-threaded event loop** — Instead of spawning a thread per request (like Java/PHP traditionally does), Node.js uses a single thread with an event loop that handles thousands of concurrent connections.
- **Same language on both frontend and backend** — Huge productivity win.

#### Topics Covered

**Node.js Core Modules**

```javascript
// File System (fs)
const fs = require('fs');
const fsPromises = require('fs').promises;

// Synchronous (blocks the event loop — avoid in production)
const data = fs.readFileSync('./config.json', 'utf-8');

// Asynchronous (preferred — non-blocking)
fs.readFile('./config.json', 'utf-8', (err, data) => {
  if (err) throw err;
  console.log(JSON.parse(data));
});

// Modern async/await style
async function readConfig() {
  const data = await fsPromises.readFile('./config.json', 'utf-8');
  return JSON.parse(data);
}
```

```javascript
// Path module — cross-platform file paths
const path = require('path');

const filePath = path.join(__dirname, 'uploads', 'image.png');
const ext = path.extname('report.pdf'); // '.pdf'
const base = path.basename('/home/user/file.txt'); // 'file.txt'
const dir = path.dirname('/home/user/file.txt'); // '/home/user'
```

```javascript
// HTTP module — building a raw server (no frameworks)
const http = require('http');

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ message: 'Hello from Node.js!' }));
});

server.listen(3000, () => {
  console.log('Server running at http://localhost:3000');
});
```

**Event Emitter Pattern**

```javascript
const EventEmitter = require('events');

class UserService extends EventEmitter {
  createUser(userData) {
    // ... create user in DB ...
    const newUser = { id: 1, ...userData };
    this.emit('user:created', newUser); // Emit an event
    return newUser;
  }
}

const userService = new UserService();

// Listen to the event
userService.on('user:created', (user) => {
  console.log(`Welcome email sent to ${user.email}`);
  // Could also trigger analytics, logging, etc.
});

userService.createUser({ name: 'Alice', email: 'alice@example.com' });
```

**Streams**

```javascript
const fs = require('fs');

// Stream a large file instead of reading it all into memory at once
const readStream = fs.createReadStream('./large-video.mp4');
const writeStream = fs.createWriteStream('./copy.mp4');

readStream.pipe(writeStream); // Elegant!

readStream.on('data', (chunk) => {
  console.log(`Received ${chunk.length} bytes`);
});

readStream.on('end', () => {
  console.log('File copy complete');
});
```

**npm & package.json**

```json
{
  "name": "my-mern-app",
  "version": "1.0.0",
  "description": "Full-stack MERN application",
  "main": "index.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js",
    "test": "jest --coverage",
    "lint": "eslint src/**/*.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "mongoose": "^8.0.0",
    "dotenv": "^16.0.0",
    "bcryptjs": "^2.4.3",
    "jsonwebtoken": "^9.0.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.0",
    "jest": "^29.0.0",
    "eslint": "^8.0.0"
  }
}
```

**Environment Variables**

```javascript
// .env file (NEVER commit this to Git!)
PORT=5000
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/mydb
JWT_SECRET=your-super-secret-key-here
NODE_ENV=development

// Loading env vars with dotenv
require('dotenv').config();
const port = process.env.PORT || 3000;
const dbUri = process.env.MONGODB_URI;
```

---

### 4. Express Fundamentals

> **Folder:** `Express Fundamentals/`  
> **Estimated Time:** 1 week

Express.js is a **minimal and flexible Node.js web framework** that provides a robust set of features for building web and mobile applications — specifically REST APIs.

#### Topics Covered

**Basic Express Server**

```javascript
const express = require('express');
const app = express();

// Built-in middleware
app.use(express.json());           // Parse JSON request bodies
app.use(express.urlencoded({ extended: true })); // Parse form data

// A simple route
app.get('/', (req, res) => {
  res.status(200).json({ message: 'API is running' });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server on port ${PORT}`));
```

**Routing**

```javascript
const router = express.Router();

// Route parameters
router.get('/users/:id', async (req, res) => {
  const { id } = req.params;
  const user = await User.findById(id);
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json(user);
});

// Query strings
router.get('/users', async (req, res) => {
  const { page = 1, limit = 10, search } = req.query;
  const query = search ? { name: { $regex: search, $options: 'i' } } : {};
  const users = await User.find(query)
    .skip((page - 1) * limit)
    .limit(Number(limit));
  res.json(users);
});

// POST with body
router.post('/users', async (req, res) => {
  const { name, email, password } = req.body;
  const user = await User.create({ name, email, password });
  res.status(201).json(user);
});
```

**Middleware — The Heart of Express**

Middleware functions are functions that have access to the `req` object, the `res` object, and the `next` function.

```javascript
// ─── Application-Level Middleware ───────────────────────────────
// Runs on every request
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next(); // Must call next() to pass control to the next middleware
});

// ─── Authentication Middleware ───────────────────────────────────
const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No token provided' });
    }
    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await User.findById(decoded.id).select('-password');
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Protect specific routes
app.get('/api/profile', authenticate, (req, res) => {
  res.json(req.user);
});

// ─── Error-Handling Middleware ──────────────────────────────────
// Must have 4 parameters to be recognized as error middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    error: err.message || 'Internal Server Error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
});
```

**CORS Configuration**

```javascript
const cors = require('cors');

const corsOptions = {
  origin: process.env.NODE_ENV === 'production'
    ? ['https://myapp.com', 'https://www.myapp.com']
    : 'http://localhost:3000',
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true, // Allow cookies
  maxAge: 86400, // Cache preflight for 24 hours
};

app.use(cors(corsOptions));
```

**Professional Project Structure**

```
backend/
├── src/
│   ├── config/
│   │   ├── db.js          # Database connection
│   │   └── env.js         # Environment variable validation
│   ├── controllers/
│   │   ├── auth.controller.js
│   │   └── user.controller.js
│   ├── middleware/
│   │   ├── auth.middleware.js
│   │   ├── error.middleware.js
│   │   └── validate.middleware.js
│   ├── models/
│   │   └── user.model.js
│   ├── routes/
│   │   ├── auth.routes.js
│   │   └── user.routes.js
│   ├── services/
│   │   └── email.service.js
│   ├── utils/
│   │   └── apiError.js
│   └── server.js
├── .env
├── .env.example
├── .gitignore
└── package.json
```

---

### 5. NoSQL Fundamentals (MongoDB)

> **Folder:** `No SQL Fundamentals/`  
> **Estimated Time:** 1 week

MongoDB is a **document-oriented NoSQL database** that stores data as flexible, JSON-like documents (called BSON — Binary JSON). It's the "M" in MERN.

#### SQL vs NoSQL

| Feature | SQL (PostgreSQL, MySQL) | NoSQL (MongoDB) |
|---------|------------------------|-----------------|
| Data Model | Tables with rows/columns | Collections with documents |
| Schema | Rigid, predefined | Flexible, dynamic |
| Relationships | JOIN operations | Embedding or referencing |
| Query Language | SQL | MongoDB Query Language (MQL) |
| Scaling | Vertical (scale up) | Horizontal (scale out) |
| Best For | Complex transactions, reporting | Flexible data, high velocity |

#### MongoDB Core Concepts

```
Database
  └── Collection (like a table)
        └── Document (like a row)
              └── Field (like a column)
```

**Example Document:**
```json
{
  "_id": "ObjectId('64a7f2c9e4b0a1234567890a')",
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "age": 28,
  "address": {
    "street": "123 Main St",
    "city": "San Francisco",
    "zip": "94102"
  },
  "tags": ["developer", "mongodb", "react"],
  "createdAt": "ISODate('2024-01-15T10:30:00.000Z')"
}
```

#### Mongoose ODM

Mongoose is an **Object Data Modeling (ODM) library** for MongoDB and Node.js. It provides schema validation, middleware hooks, and a clean API.

**Defining a Schema & Model:**

```javascript
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: [true, 'Name is required'],
      trim: true,
      minlength: [2, 'Name must be at least 2 characters'],
      maxlength: [50, 'Name cannot exceed 50 characters'],
    },
    email: {
      type: String,
      required: [true, 'Email is required'],
      unique: true,
      lowercase: true,
      match: [/^\S+@\S+\.\S+$/, 'Please enter a valid email'],
    },
    password: {
      type: String,
      required: true,
      minlength: 8,
      select: false, // Never returned in queries by default
    },
    role: {
      type: String,
      enum: ['user', 'admin', 'moderator'],
      default: 'user',
    },
    isActive: { type: Boolean, default: true },
    lastLogin: Date,
  },
  {
    timestamps: true, // Adds createdAt and updatedAt automatically
    toJSON: { virtuals: true },
  }
);

// ─── Virtual Field ──────────────────────────────────────────────
userSchema.virtual('fullProfile').get(function () {
  return `${this.name} <${this.email}>`;
});

// ─── Pre-save Hook (Middleware) ─────────────────────────────────
userSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();
  const salt = await bcrypt.genSalt(12);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// ─── Instance Method ────────────────────────────────────────────
userSchema.methods.comparePassword = async function (candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

// ─── Static Method ──────────────────────────────────────────────
userSchema.statics.findByEmail = function (email) {
  return this.findOne({ email: email.toLowerCase() });
};

// ─── Index for Performance ──────────────────────────────────────
userSchema.index({ email: 1 });
userSchema.index({ createdAt: -1 });

const User = mongoose.model('User', userSchema);
module.exports = User;
```

**CRUD Operations with Mongoose:**

```javascript
// ─── CREATE ─────────────────────────────────────────────────────
const user = new User({ name: 'Alice', email: 'alice@example.com', password: 'secret123' });
await user.save();
// OR
const user = await User.create({ name: 'Alice', email: 'alice@example.com', password: 'secret123' });

// ─── READ ───────────────────────────────────────────────────────
const allUsers = await User.find();
const activeAdmins = await User.find({ role: 'admin', isActive: true });
const user = await User.findById(id);
const user = await User.findOne({ email: 'alice@example.com' });

// With field selection, sorting, pagination
const users = await User.find({ isActive: true })
  .select('name email role createdAt')  // Only these fields
  .sort({ createdAt: -1 })              // Newest first
  .skip(0)                              // Pagination offset
  .limit(10)                            // Page size
  .populate('posts', 'title');          // Join related data

// ─── UPDATE ─────────────────────────────────────────────────────
await User.findByIdAndUpdate(id, { name: 'Alice Smith' }, { new: true, runValidators: true });
await User.updateMany({ role: 'user' }, { $set: { isActive: true } });

// ─── DELETE ─────────────────────────────────────────────────────
await User.findByIdAndDelete(id);
await User.deleteMany({ isActive: false });
```

**Aggregation Pipeline:**

```javascript
// Powerful data transformation — like SQL GROUP BY on steroids
const stats = await Order.aggregate([
  { $match: { status: 'completed', createdAt: { $gte: new Date('2024-01-01') } } },
  { $group: {
      _id: '$userId',
      totalOrders: { $sum: 1 },
      totalSpent: { $sum: '$amount' },
      avgOrder: { $avg: '$amount' },
  }},
  { $sort: { totalSpent: -1 } },
  { $limit: 10 },
  { $lookup: {
      from: 'users',
      localField: '_id',
      foreignField: '_id',
      as: 'user',
  }},
  { $unwind: '$user' },
  { $project: {
      'user.name': 1,
      'user.email': 1,
      totalOrders: 1,
      totalSpent: 1,
  }},
]);
```

---

### 6. SQL Fundamentals

> **Folder:** `SQL Fundamentals/`  
> **Estimated Time:** 1 week

While MERN traditionally uses MongoDB, knowing SQL is critical for many real-world projects and interviews. This module covers relational databases.

#### Core SQL Commands

```sql
-- CREATE TABLE
CREATE TABLE users (
  id         SERIAL PRIMARY KEY,
  name       VARCHAR(100) NOT NULL,
  email      VARCHAR(255) UNIQUE NOT NULL,
  role       VARCHAR(20) DEFAULT 'user',
  created_at TIMESTAMP DEFAULT NOW()
);

-- INSERT
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');

-- SELECT with filtering, ordering, pagination
SELECT name, email, created_at
FROM users
WHERE role = 'admin' AND created_at > '2024-01-01'
ORDER BY created_at DESC
LIMIT 10 OFFSET 20;

-- JOIN
SELECT u.name, p.title, p.created_at
FROM users u
INNER JOIN posts p ON p.user_id = u.id
WHERE u.id = 42;

-- GROUP BY with aggregation
SELECT
  DATE_TRUNC('month', created_at) AS month,
  COUNT(*) AS new_users
FROM users
GROUP BY month
ORDER BY month;

-- UPDATE & DELETE
UPDATE users SET role = 'admin' WHERE email = 'alice@example.com';
DELETE FROM users WHERE last_login < NOW() - INTERVAL '1 year';
```

---

### 7. React Fundamentals

> **Folder:** `React Fundamentals/`  
> **Language:** JavaScript/JSX  
> **Estimated Time:** 2 weeks

React is a **declarative, component-based JavaScript library** for building user interfaces. Created by Facebook/Meta, it is the most widely used frontend framework.

#### Core Philosophy

React is built around one central idea: **UI as a function of state**.

```
UI = f(state)
```

When state changes, React efficiently re-renders only the parts of the UI that need to change (via the Virtual DOM diffing algorithm).

#### Topics Covered

**JSX — JavaScript + HTML**

```jsx
// JSX lets you write HTML-like syntax in JavaScript
// Under the hood, JSX transpiles to React.createElement() calls

const greeting = <h1 className="title">Hello, {name}!</h1>;

// Expressions in JSX
const element = (
  <div>
    <p>{isLoggedIn ? 'Welcome back!' : 'Please sign in'}</p>
    <ul>
      {items.map(item => (
        <li key={item.id}>{item.name}</li>  // key prop is required in lists!
      ))}
    </ul>
  </div>
);
```

**Functional Components & Props**

```jsx
// A simple presentational component
function UserCard({ name, email, avatarUrl, onFollow }) {
  return (
    <div className="card">
      <img src={avatarUrl} alt={name} />
      <h3>{name}</h3>
      <p>{email}</p>
      <button onClick={onFollow}>Follow</button>
    </div>
  );
}

// Usage
<UserCard
  name="Alice"
  email="alice@example.com"
  avatarUrl="/avatars/alice.jpg"
  onFollow={() => handleFollow('alice')}
/>
```

**React Hooks**

```jsx
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

function UserList() {
  // ─── useState: Managing local component state ──────────────────
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);

  // ─── useRef: Mutable value that doesn't trigger re-render ──────
  const abortControllerRef = useRef(null);

  // ─── useEffect: Side effects (data fetching, subscriptions) ───
  useEffect(() => {
    const fetchUsers = async () => {
      // Cancel previous request if component re-renders quickly
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();

      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `/api/users?page=${page}&search=${searchTerm}`,
          { signal: abortControllerRef.current.signal }
        );
        const data = await res.json();
        setUsers(data.users);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();

    // Cleanup function runs before next effect or on unmount
    return () => abortControllerRef.current?.abort();
  }, [page, searchTerm]); // Re-run when these dependencies change

  // ─── useCallback: Memoize functions ───────────────────────────
  const handleDelete = useCallback(async (userId) => {
    await fetch(`/api/users/${userId}`, { method: 'DELETE' });
    setUsers(prev => prev.filter(u => u.id !== userId));
  }, []);

  // ─── useMemo: Expensive computations ──────────────────────────
  const sortedUsers = useMemo(
    () => [...users].sort((a, b) => a.name.localeCompare(b.name)),
    [users]
  );

  if (loading) return <div className="spinner" />;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div>
      <input value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
      {sortedUsers.map(user => (
        <UserCard key={user.id} {...user} onDelete={() => handleDelete(user.id)} />
      ))}
    </div>
  );
}
```

**Custom Hooks — Reusing Logic**

```jsx
// A custom hook that encapsulates fetch logic
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    fetch(url)
      .then(res => res.json())
      .then(json => { if (isMounted) setData(json); })
      .catch(err => { if (isMounted) setError(err.message); })
      .finally(() => { if (isMounted) setLoading(false); });

    return () => { isMounted = false; };
  }, [url]);

  return { data, loading, error };
}

// Usage in any component — clean and reusable!
function ProductPage({ productId }) {
  const { data: product, loading, error } = useFetch(`/api/products/${productId}`);

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} />;
  return <ProductDetail product={product} />;
}
```

**Context API — Global State**

```jsx
// ─── Creating a Context ─────────────────────────────────────────
const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing session on mount
    const token = localStorage.getItem('token');
    if (token) {
      verifyToken(token).then(setUser).finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (credentials) => {
    const { user, token } = await loginApi(credentials);
    localStorage.setItem('token', token);
    setUser(user);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to consume context
export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};
```

**React Router v6**

```jsx
import { BrowserRouter, Routes, Route, Navigate, useParams } from 'react-router-dom';

function App() {
  const { user } = useAuth();

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/products" element={<ProductListPage />} />
        <Route path="/products/:id" element={<ProductDetailPage />} />
        <Route
          path="/dashboard/*"
          element={user ? <DashboardLayout /> : <Navigate to="/login" replace />}
        >
          <Route index element={<DashboardHome />} />
          <Route path="profile" element={<ProfilePage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </BrowserRouter>
  );
}
```

---

### 8. Next.js Fundamentals

> **Folder:** `NextJS Fundamentals/`  
> **Estimated Time:** 1 week

Next.js is a **full-stack React framework** built by Vercel. It adds SSR, SSG, file-based routing, API routes, and much more on top of React.

#### Rendering Strategies

| Strategy | When HTML is generated | Use Case |
|----------|----------------------|----------|
| **CSR** (Client-Side Rendering) | In the browser at runtime | Highly interactive dashboards |
| **SSR** (Server-Side Rendering) | On the server per request | SEO-critical, personalized pages |
| **SSG** (Static Site Generation) | At build time | Blogs, marketing pages |
| **ISR** (Incremental Static Regeneration) | Build time + on-demand revalidation | E-commerce product pages |

#### App Router (Next.js 13+)

```
app/
├── layout.tsx           # Root layout (wraps all pages)
├── page.tsx             # / (homepage)
├── loading.tsx          # Automatic loading UI
├── error.tsx            # Error boundary
├── (auth)/              # Route group (doesn't affect URL)
│   ├── login/page.tsx   # /login
│   └── register/page.tsx # /register
├── dashboard/
│   ├── layout.tsx       # Dashboard-specific layout
│   └── page.tsx         # /dashboard
├── api/
│   └── users/
│       └── route.ts     # /api/users
└── products/
    ├── page.tsx         # /products
    └── [id]/
        └── page.tsx     # /products/:id
```

**Server Components vs. Client Components:**

```tsx
// Server Component (default) — runs on the server
// Can directly query database, access filesystem, use API keys
async function ProductList() {
  const products = await db.product.findMany(); // Direct DB access!
  return (
    <ul>
      {products.map(p => <ProductCard key={p.id} product={p} />)}
    </ul>
  );
}

// Client Component — runs in the browser
'use client';
import { useState } from 'react';

function AddToCartButton({ productId }) {
  const [added, setAdded] = useState(false);

  return (
    <button onClick={() => { addToCart(productId); setAdded(true); }}>
      {added ? '✓ Added!' : 'Add to Cart'}
    </button>
  );
}
```

---

### 9. NestJS Fundamentals

> **Folder:** `NestJS Fundamentals/`  
> **Estimated Time:** 1 week

NestJS is a **progressive Node.js framework** for building efficient and scalable server-side applications. It uses TypeScript by default and is heavily inspired by Angular's architecture.

#### Key Concepts

```
┌─────────────────────────────────────┐
│            NestJS Architecture       │
│                                      │
│  Request → Guards → Interceptors     │
│    → Pipes → Controller → Service    │
│      → Repository → Database        │
└─────────────────────────────────────┘
```

```typescript
// users.module.ts
@Module({
  imports: [TypeOrmModule.forFeature([User])],
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService],
})
export class UsersModule {}

// users.controller.ts
@Controller('users')
@UseGuards(JwtAuthGuard)
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Get()
  @Roles('admin')
  findAll(@Query() paginationDto: PaginationDto) {
    return this.usersService.findAll(paginationDto);
  }

  @Get(':id')
  findOne(@Param('id', ParseIntPipe) id: number) {
    return this.usersService.findOne(id);
  }

  @Post()
  @HttpCode(HttpStatus.CREATED)
  create(@Body() createUserDto: CreateUserDto) {
    return this.usersService.create(createUserDto);
  }
}
```

---

### 10. React Native Fundamentals

> **Folder:** `React Native Fundamentals/`  
> **Estimated Time:** 1 week

React Native lets you build **native iOS and Android apps** using React. If you know React, you already know 80% of React Native.

```jsx
// React Web vs React Native
// Web:   <div>     → Native: <View>
// Web:   <p>       → Native: <Text>
// Web:   <img>     → Native: <Image>
// Web:   <button>  → Native: <TouchableOpacity> or <Pressable>

import { View, Text, FlatList, StyleSheet, TouchableOpacity } from 'react-native';

function UserListScreen({ navigation }) {
  const [users, setUsers] = useState([]);

  return (
    <View style={styles.container}>
      <FlatList
        data={users}
        keyExtractor={item => item.id.toString()}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.item}
            onPress={() => navigation.navigate('UserDetail', { userId: item.id })}
          >
            <Text style={styles.name}>{item.name}</Text>
            <Text style={styles.email}>{item.email}</Text>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  item: { padding: 16, borderBottomWidth: 1, borderBottomColor: '#eee' },
  name: { fontSize: 16, fontWeight: '600' },
  email: { fontSize: 14, color: '#666' },
});
```

---

### 11. Vue.js Fundamentals

> **Folder:** `vueJS Fundamentals/`  
> **Estimated Time:** 3–4 days

Vue.js is a **progressive JavaScript framework** for building user interfaces. It's an excellent alternative to React with a gentler learning curve.

```vue
<!-- Single File Component (SFC) -->
<template>
  <div class="user-card">
    <h2>{{ user.name }}</h2>
    <p v-if="user.isActive" class="badge">Active</p>
    <ul>
      <li v-for="post in user.posts" :key="post.id">
        {{ post.title }}
      </li>
    </ul>
    <button @click="followUser">Follow</button>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';

const props = defineProps({ userId: String });
const user = ref(null);
const isFollowing = ref(false);

const followerCount = computed(() =>
  user.value?.followers.length ?? 0
);

onMounted(async () => {
  user.value = await fetchUser(props.userId);
});

const followUser = () => {
  isFollowing.value = !isFollowing.value;
};
</script>

<style scoped>
.badge { color: green; font-weight: bold; }
</style>
```

---

### 12. Bun.js Fundamentals

> **Folder:** `bunJS Fundamentals/`  
> **Estimated Time:** 2–3 days

Bun is a **fast all-in-one JavaScript runtime** (like Node.js but much faster), built from scratch using the JavaScriptCore engine (instead of V8). It's also a package manager, bundler, and test runner.

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash

# Bun is 3-5x faster than Node for many tasks
bun run server.ts    # Run TypeScript natively — no transpilation step!
bun install          # Install packages (faster than npm/yarn/pnpm)
bun test             # Run tests with built-in test runner
bun build ./src/index.ts --outdir ./dist  # Bundle your app
```

```typescript
// Bun's built-in HTTP server
Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === '/api/users') {
      const users = await db.query('SELECT * FROM users');
      return Response.json(users);
    }

    return new Response('Not Found', { status: 404 });
  },
});
```

---

### 13. NGINX Fundamentals

> **Folder:** `NGINX Fundamentals/`  
> **Estimated Time:** 3–4 days

NGINX (pronounced "engine-x") is a high-performance **web server, reverse proxy, load balancer, and HTTP cache**. In MERN deployments, NGINX typically sits in front of your Node.js server.

#### MERN Production Architecture with NGINX

```
                    Internet
                       │
                       ▼
              ┌─────────────────┐
              │   NGINX (80/443) │  ← Handles SSL, serves static files
              └─────────┬───────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
  ┌──────────────┐          ┌──────────────────┐
  │ React Build  │          │  Node.js API     │
  │ (/var/www)   │          │  (port 5000)     │
  └──────────────┘          └──────────────────┘
                                      │
                              ┌───────▼───────┐
                              │   MongoDB     │
                              └───────────────┘
```

**NGINX Config for MERN:**

```nginx
# /etc/nginx/sites-available/myapp.conf

server {
    listen 80;
    server_name myapp.com www.myapp.com;
    return 301 https://$host$request_uri;  # Redirect HTTP to HTTPS
}

server {
    listen 443 ssl http2;
    server_name myapp.com www.myapp.com;

    ssl_certificate /etc/letsencrypt/live/myapp.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/myapp.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";
    add_header Strict-Transport-Security "max-age=63072000";

    # Serve React build (static files)
    root /var/www/myapp/client/build;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json;

    # Reverse proxy to Express API
    location /api/ {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
    }

    # React SPA fallback — critical for client-side routing
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

### 14. Full MERN Stack Project

> **File:** `MernStack`  
> **Estimated Time:** 1 week

This is the **capstone project** that ties everything together — a full-stack MERN application with authentication, CRUD operations, file uploads, and deployment.

#### Project Architecture

```
mern-app/
├── client/                    # React frontend
│   ├── public/
│   ├── src/
│   │   ├── api/               # Axios/fetch service layer
│   │   ├── components/        # Reusable UI components
│   │   ├── context/           # Auth context, theme context
│   │   ├── hooks/             # Custom React hooks
│   │   ├── pages/             # Route-level components
│   │   ├── utils/             # Helper functions
│   │   └── App.jsx
│   └── package.json
│
├── server/                    # Express backend
│   ├── src/
│   │   ├── config/
│   │   ├── controllers/
│   │   ├── middleware/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── services/
│   │   └── server.js
│   └── package.json
│
├── docker-compose.yml
└── README.md
```

#### JWT Authentication Flow

```
1. User submits login form (POST /api/auth/login)
2. Server validates credentials against MongoDB
3. Server signs a JWT with user ID and role (expires in 7d)
4. Client stores JWT in memory (or httpOnly cookie)
5. Client sends JWT in Authorization header for protected routes
6. Server middleware validates JWT on each protected request
7. If valid → req.user is populated → route handler runs
8. If invalid/expired → 401 Unauthorized response
```

```javascript
// JWT Generation
const token = jwt.sign(
  { id: user._id, role: user.role },
  process.env.JWT_SECRET,
  { expiresIn: '7d' }
);

// JWT Verification Middleware
const authenticate = async (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await User.findById(decoded.id).select('-password');
    next();
  } catch {
    res.status(401).json({ error: 'Token invalid or expired' });
  }
};
```

---

## 🚀 Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/CoderNived/MernStack.git
cd MernStack
```

### Step 2: Install MongoDB

**Option A: MongoDB Atlas (Cloud — Recommended for beginners)**
1. Go to [https://cloud.mongodb.com](https://cloud.mongodb.com)
2. Create a free cluster
3. Whitelist your IP address
4. Get your connection string: `mongodb+srv://username:password@cluster.mongodb.net/dbname`

**Option B: Local MongoDB**
```bash
# macOS (using Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Ubuntu
sudo apt-get install -y mongodb
sudo systemctl start mongodb

# Verify it's running
mongo --version
```

### Step 3: Set Up Each Module

```bash
# Navigate to a module
cd "NodeJS Fundamentals"
npm install
npm run dev

# Or for a React module
cd "../React Fundamentals"
npm install
npm start
```

### Step 4: Environment Variables

Each module that requires environment variables will have a `.env.example` file. Copy it and fill in your values:

```bash
cp .env.example .env
# Edit .env with your actual values
```

---

## 💡 Core Concepts Explained

### The Event Loop (Node.js)

```
┌─────────────────────────────────────────┐
│                                          │
│   Call Stack    │   Callback Queue       │
│   ─────────     │   ──────────────       │
│   main()        │   setTimeout cb        │
│   readFile()    │   click handler        │
│                 │                        │
│         Event Loop                       │
│    (moves callbacks from queue to        │
│     call stack when stack is empty)      │
│                                          │
│   Web APIs / Node APIs                   │
│   ─────────────────────                  │
│   setTimeout, fetch, fs.readFile         │
│   (These run outside the main thread)    │
└─────────────────────────────────────────┘
```

### React's Virtual DOM

```
1. State changes in a component
2. React creates a new Virtual DOM tree (in-memory JS representation)
3. React diffs the new tree against the previous (reconciliation)
4. Only the changed DOM nodes are updated in the real DOM
5. This batched, minimal DOM update is what makes React fast
```

### MongoDB Document Relationships

```javascript
// ─── Embedding (for 1:few relationships) ───────────────────────
// Best when: sub-documents are always read with the parent
// and there aren't too many of them
{
  _id: ObjectId("..."),
  name: "Alice",
  addresses: [                     // Embedded array
    { type: "home", city: "NYC" },
    { type: "work", city: "SF" }
  ]
}

// ─── Referencing (for 1:many or many:many) ─────────────────────
// Best when: sub-documents are large, frequently updated independently,
// or need to be accessed on their own
{
  _id: ObjectId("..."),
  title: "My Post",
  author: ObjectId("507f1f77bcf86cd799439011")  // Reference to User
}

// Use .populate() in Mongoose to join
const post = await Post.findById(id).populate('author', 'name email');
```

---

## 🏆 Best Practices & Patterns

### Security Checklist

```bash
# ─── Input Validation ──────────────────────────────────────────
npm install joi          # Schema validation
npm install express-validator  # Middleware-based validation

# ─── Authentication & Authorization ───────────────────────────
npm install bcryptjs     # Password hashing (NEVER store plain text)
npm install jsonwebtoken # JWT tokens

# ─── Security Middleware ───────────────────────────────────────
npm install helmet       # Sets secure HTTP headers
npm install express-rate-limit  # Prevent brute force attacks
npm install express-mongo-sanitize  # Prevent NoSQL injection
npm install xss-clean    # Prevent XSS attacks
npm install hpp          # HTTP Parameter Pollution prevention
```

```javascript
// Apply all security middleware
app.use(helmet());
app.use(mongoSanitize());
app.use(xss());
app.use(hpp());

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Max 100 requests per window
  message: 'Too many requests, please try again later.',
});
app.use('/api/', limiter);
```

### Error Handling Pattern

```javascript
// Custom AppError class
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true; // vs programming errors
    Error.captureStackTrace(this, this.constructor);
  }
}

// Async wrapper to avoid try/catch in every controller
const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

// Usage — clean controllers
exports.getUser = asyncHandler(async (req, res, next) => {
  const user = await User.findById(req.params.id);
  if (!user) return next(new AppError('User not found', 404));
  res.json(user);
});
```

### React Performance Patterns

```jsx
// ─── Code Splitting with lazy loading ─────────────────────────
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/admin" element={<AdminDashboard />} />
      </Routes>
    </Suspense>
  );
}

// ─── Memoization ──────────────────────────────────────────────
// React.memo prevents re-render if props haven't changed
const UserCard = memo(function UserCard({ user, onDelete }) {
  console.log('UserCard rendering:', user.id);
  return (/* ... */);
});
```

---

## 🚢 Deployment Guide

### Docker Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  client:
    build: ./client
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://api:5000

  api:
    build: ./server
    ports:
      - "5000:5000"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/myapp
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - mongo

  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

### Production Deployment Options

| Platform | Best For | Free Tier |
|----------|----------|-----------|
| **Vercel** | Next.js / React frontend | ✅ Yes |
| **Railway** | Full-stack Node.js apps | ✅ Yes |
| **Render** | Web services + DBs | ✅ Yes |
| **Heroku** | Simple deployment | ❌ No longer |
| **AWS EC2 + Nginx** | Full control, production | ⚠️ Micro instance |
| **DigitalOcean Droplet** | VPS with full control | ⚠️ $4/mo |

---

## 🔧 Troubleshooting

### Common Issues & Solutions

**MongoDB Connection Error**
```
MongoNetworkError: failed to connect to server
```
→ Check your `MONGODB_URI` in `.env`  
→ Whitelist your IP in MongoDB Atlas  
→ Ensure `mongod` is running locally  

**CORS Error**
```
Access to fetch blocked by CORS policy
```
→ Ensure your Express server has `cors()` middleware  
→ Check the `origin` option matches your frontend URL  
→ Don't forget `credentials: true` if using cookies  

**React "Cannot read properties of undefined"**
```
TypeError: Cannot read properties of undefined (reading 'map')
```
→ Your data hasn't loaded yet. Initialize state as an empty array: `useState([])`  
→ Use optional chaining: `data?.users.map(...)`  

**JWT "invalid signature"**
→ Make sure you use the same `JWT_SECRET` to sign and verify  
→ Ensure you're not accidentally comparing different environment files  

**Port already in use**
```bash
# Find and kill the process using the port
lsof -ti:3000 | xargs kill -9   # macOS/Linux
netstat -ano | findstr :3000    # Windows
```

---

## 🤝 Contributing

Contributions are welcome! If you find a bug, have a suggestion, or want to add a new example, please:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/add-redis-example`
3. Commit your changes: `git commit -m "feat: add Redis caching example to Express module"`
4. Push to your fork: `git push origin feature/add-redis-example`
5. Open a **Pull Request** with a clear description of your changes

### Commit Message Convention

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat:     New example or module added
fix:      Bug fix in existing code
docs:     Documentation update
refactor: Code restructuring (no behavior change)
chore:    Build process or config changes
```

---

## 📚 Additional Resources

### Official Documentation
- [Node.js Docs](https://nodejs.org/en/docs/)
- [Express.js Guide](https://expressjs.com/en/guide/routing.html)
- [MongoDB Manual](https://www.mongodb.com/docs/manual/)
- [Mongoose Docs](https://mongoosejs.com/docs/)
- [React Docs](https://react.dev/)
- [Next.js Docs](https://nextjs.org/docs)
- [NestJS Docs](https://docs.nestjs.com/)

### Books
- *You Don't Know JS* — Kyle Simpson (free on GitHub)
- *Node.js Design Patterns* — Mario Casciaro
- *Learning React* — Alex Banks & Eve Porcello

### Courses & Practice
- [The Odin Project](https://www.theodinproject.com/) — Free full-stack curriculum
- [FreeCodeCamp](https://www.freecodecamp.org/) — Hands-on coding challenges
- [Frontend Mentor](https://www.frontendmentor.io/) — Real-world UI challenges

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for the developer community**

If this repository helped you, please give it a ⭐ on GitHub!

[Back to Top](#-mern-stack--complete-full-stack-development-curriculum)

</div>
