import 'dotenv/config';
import dns from 'node:dns';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import nodemailer from 'nodemailer';

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });
const PORT = Number(process.env.PORT || 5000);
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://127.0.0.1:8000';
const JWT_SECRET = process.env.JWT_SECRET || 'development-only-change-this-secret';

// Some Windows/router DNS resolvers reject MongoDB Atlas SRV lookups even
// though nslookup can resolve them. Use configurable public DNS for Node's
// SRV resolver; set MONGODB_DNS_SERVERS to override this in restricted networks.
const dnsServers = (process.env.MONGODB_DNS_SERVERS || '8.8.8.8,1.1.1.1').split(',').map(value => value.trim()).filter(Boolean);
try { dns.setServers(dnsServers); } catch (error) { console.warn('Could not configure Node DNS servers:', error.message); }

app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: '2mb' }));

const userSchema = new mongoose.Schema({ username: { type: String, unique: true, required: true }, email: { type: String, unique: true, sparse: true }, display_name: String, password_hash: { type: String, required: true }, role: { type: String, default: 'doctor' }, active: { type: Boolean, default: true }, reset_token: String, reset_expires_at: Date }, { timestamps: true });
const patientSchema = new mongoose.Schema({ patient_id: { type: String, unique: true }, name: String, age: Number, gender: String, history: String, symptoms: String, referring_doctor: String, location: String }, { timestamps: true });
const scanSchema = new mongoose.Schema({ patient_id: String, filename: String, prediction: String, confidence: Number, probabilities: mongoose.Schema.Types.Mixed, metadata: mongoose.Schema.Types.Mixed, image_preview: String }, { timestamps: true });
const reportSchema = new mongoose.Schema({ patient_id: String, scan_id: String, doctor_name: String, decision: String, notes: String }, { timestamps: true });
const auditSchema = new mongoose.Schema({ user_id: String, action: String, entity_type: String, entity_id: String, details: String }, { timestamps: true });
const User = mongoose.model('User', userSchema);
const Patient = mongoose.model('Patient', patientSchema);
const Scan = mongoose.model('Scan', scanSchema);
const Report = mongoose.model('Report', reportSchema);
const AuditEvent = mongoose.model('AuditEvent', auditSchema);

const publicUser = user => ({ id: user._id.toString(), username: user.username, email: user.email, display_name: user.display_name, role: user.role, active: user.active !== false, created_at: user.createdAt });
const tokenFor = user => jwt.sign({ id: user._id.toString() }, JWT_SECRET, { expiresIn: '7d' });

async function audit(user, action, entity_type, entity_id = '', details = '') {
  await AuditEvent.create({ user_id: user?._id?.toString(), action, entity_type, entity_id, details });
}

function requireAuth(req, res, next) {
  try {
    const header = req.headers.authorization || '';
    if (!header.startsWith('Bearer ')) return res.status(401).json({ detail: 'Authentication required' });
    const decoded = jwt.verify(header.slice(7), JWT_SECRET);
    User.findById(decoded.id).then(user => {
      if (!user || user.active === false) return res.status(401).json({ detail: 'Invalid or expired session' });
      req.user = user;
      next();
    }).catch(() => res.status(401).json({ detail: 'Invalid or expired session' }));
  } catch {
    return res.status(401).json({ detail: 'Invalid or expired session' });
  }
}

function requireRole(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user?.role)) return res.status(403).json({ detail: 'This action requires an authorized clinical role.' });
    next();
  };
}

async function sendEmail(recipient, subject, body) {
  if (!process.env.SMTP_HOST) return { status: 'not_configured', message: 'SMTP_HOST is not configured; report was saved but not emailed.' };
  const transporter = nodemailer.createTransport({ host: process.env.SMTP_HOST, port: Number(process.env.SMTP_PORT || 587), secure: Number(process.env.SMTP_PORT || 587) === 465, auth: process.env.SMTP_USERNAME ? { user: process.env.SMTP_USERNAME, pass: process.env.SMTP_PASSWORD } : undefined, requireTLS: process.env.SMTP_USE_TLS !== 'false' });
  await transporter.sendMail({ from: process.env.SMTP_FROM || process.env.SMTP_USERNAME, to: recipient, subject, text: body });
  return { status: 'sent', message: `Report emailed to ${recipient}.` };
}

app.get('/', (req, res) => res.json({ status: 'online', system: 'TumorGANet Express API', ai_service: AI_SERVICE_URL }));

app.post('/api/auth/login', async (req, res) => {
  const username = String(req.body.username || '').trim().toLowerCase();
  const user = await User.findOne({ $or: [{ username }, { email: username }] });
  if (!user || user.active === false || !(await bcrypt.compare(String(req.body.password || ''), user.password_hash))) return res.status(401).json({ detail: user?.active === false ? 'This account is disabled. Contact an administrator.' : 'Invalid username or password' });
  await audit(user, 'login', 'session', '', 'User signed in');
  res.json({ access_token: tokenFor(user), user: publicUser(user) });
});

app.post('/api/auth/signup', async (req, res) => {
  try {
    const email = String(req.body.email || '').trim().toLowerCase();
    if (String(req.body.password || '').length < 8) return res.status(400).json({ detail: 'Password must contain at least 8 characters' });
    const user = await User.create({ username: email, email, display_name: String(req.body.name || email).trim(), password_hash: await bcrypt.hash(req.body.password, 12), role: 'doctor' });
    await audit(user, 'signup', 'user', user._id.toString(), 'Account created');
    res.json({ access_token: tokenFor(user), user: publicUser(user) });
  } catch (error) {
    res.status(400).json({ detail: error.code === 11000 ? 'An account with this email already exists' : error.message });
  }
});

app.get('/api/auth/me', requireAuth, (req, res) => res.json(publicUser(req.user)));
app.post('/api/auth/logout', requireAuth, async (req, res) => { await audit(req.user, 'logout', 'session', '', 'User signed out'); res.json({ message: 'Signed out' }); });

app.post('/api/auth/forgot-password', async (req, res) => {
  const user = await User.findOne({ email: String(req.body.email || '').trim().toLowerCase() });
  if (!user) return res.json({ message: 'If an account exists for that email, reset instructions have been sent.' });
  const resetToken = jwt.sign({ purpose: 'reset', id: user._id.toString() }, JWT_SECRET, { expiresIn: '1h' });
  user.reset_token = resetToken;
  user.reset_expires_at = new Date(Date.now() + 3600000);
  await user.save();
  const delivery = await sendEmail(user.email, 'TumorGANet password reset', `Use this password reset token within one hour:\n\n${resetToken}`);
  res.json({ message: 'If an account exists for that email, reset instructions have been sent.', ...(delivery.status === 'not_configured' ? { development_token: resetToken } : {}) });
});

app.post('/api/auth/reset-password', async (req, res) => {
  const user = await User.findOne({ reset_token: req.body.token, reset_expires_at: { $gt: new Date() } });
  if (!user || String(req.body.password || '').length < 8) return res.status(400).json({ detail: 'Invalid or expired reset token' });
  user.password_hash = await bcrypt.hash(req.body.password, 12); user.reset_token = undefined; user.reset_expires_at = undefined; await user.save();
  res.json({ message: 'Password reset successfully' });
});

app.get('/api/admin/users', requireAuth, requireRole('admin'), async (req, res) => {
  const users = await User.find().sort({ createdAt: -1 });
  res.json({ users: users.map(publicUser) });
});

app.post('/api/admin/users', requireAuth, requireRole('admin'), async (req, res) => {
  try {
    const email = String(req.body.email || '').trim().toLowerCase();
    const role = String(req.body.role || 'technician').trim().toLowerCase();
    const displayName = String(req.body.display_name || req.body.name || email).trim();
    const password = String(req.body.password || '');
    if (!email || !email.includes('@')) return res.status(400).json({ detail: 'A valid email is required' });
    if (!['doctor', 'technician'].includes(role)) return res.status(400).json({ detail: 'Admins can create doctor or technician accounts only' });
    if (password.length < 8) return res.status(400).json({ detail: 'Password must contain at least 8 characters' });
    const user = await User.create({ username: email, email, display_name: displayName, password_hash: await bcrypt.hash(password, 12), role, active: true });
    await audit(req.user, 'create_user', 'user', user._id.toString(), `Created ${role} account for ${email}`);
    res.status(201).json({ user: publicUser(user) });
  } catch (error) {
    res.status(400).json({ detail: error.code === 11000 ? 'An account with this email already exists' : error.message });
  }
});

app.patch('/api/admin/users/:id', requireAuth, requireRole('admin'), async (req, res) => {
  if (String(req.params.id) === req.user._id.toString()) return res.status(400).json({ detail: 'You cannot change or disable your own admin account' });
  const user = await User.findById(req.params.id);
  if (!user) return res.status(404).json({ detail: 'User not found' });
  const updates = {};
  if (req.body.role !== undefined) {
    const role = String(req.body.role).trim().toLowerCase();
    if (!['doctor', 'technician', 'admin'].includes(role)) return res.status(400).json({ detail: 'Invalid user role' });
    updates.role = role;
  }
  if (req.body.active !== undefined) updates.active = Boolean(req.body.active);
  if (!Object.keys(updates).length) return res.status(400).json({ detail: 'No valid account changes were provided' });
  Object.assign(user, updates);
  await user.save();
  await audit(req.user, 'update_user', 'user', user._id.toString(), `Updated account: ${JSON.stringify(updates)}`);
  res.json({ user: publicUser(user) });
});

app.post('/api/admin/users/:id/password', requireAuth, requireRole('admin'), async (req, res) => {
  const user = await User.findById(req.params.id);
  if (!user) return res.status(404).json({ detail: 'User not found' });
  const password = String(req.body.password || '');
  if (password.length < 8) return res.status(400).json({ detail: 'Password must contain at least 8 characters' });
  user.password_hash = await bcrypt.hash(password, 12);
  user.reset_token = undefined;
  user.reset_expires_at = undefined;
  await user.save();
  await audit(req.user, 'reset_password', 'user', user._id.toString(), `Password reset for ${user.email}`);
  res.json({ message: 'Password reset successfully' });
});

app.get('/api/admin/activity', requireAuth, requireRole('admin'), async (req, res) => {
  const query = req.query.user_id ? { user_id: String(req.query.user_id) } : {};
  const events = await AuditEvent.find(query).sort({ createdAt: -1 }).limit(200);
  res.json({ events });
});

app.get('/api/patients', requireAuth, async (req, res) => { const q = String(req.query.search || ''); const patients = await Patient.find(q ? { $or: [{ name: new RegExp(q, 'i') }, { patient_id: new RegExp(q, 'i') }] } : {}).sort({ createdAt: -1 }); res.json({ patients: patients.map(p => ({ id: p.patient_id, name: p.name, age: p.age, gender: p.gender, history: p.history, symptoms: p.symptoms, referring_doctor: p.referring_doctor, location: p.location, created_at: p.createdAt })) }); });
async function upsertPatient(payload) { const patient = payload || {}; const patientId = patient.id || `PT-${Math.floor(1000 + Math.random() * 9000)}`; return Patient.findOneAndUpdate({ patient_id: patientId }, { patient_id: patientId, name: patient.name || 'Unnamed patient', age: patient.age, gender: patient.gender || 'Not specified', history: patient.history || '', symptoms: patient.symptoms || '', referring_doctor: patient.referring_doctor || patient.referringDoctor || '', location: patient.location || '' }, { upsert: true, new: true, setDefaultsOnInsert: true }); }
app.post('/api/patients', requireAuth, async (req, res) => { const p = await upsertPatient(req.body); await audit(req.user, 'create', 'patient', p.patient_id, 'Patient record created'); res.json({ id: p.patient_id, name: p.name, age: p.age, gender: p.gender, history: p.history, symptoms: p.symptoms, referring_doctor: p.referring_doctor, location: p.location, created_at: p.createdAt }); });

app.get('/api/doctors', requireAuth, (req, res) => {
  const city = String(req.query.city || '').trim();
  const specialty = String(req.query.specialty || 'neurosurgeon').trim();
  if (!city) return res.json({ doctors: [], search_url: '', disclaimer: 'Add the patient\'s city or region to search for nearby specialists.' });

  // This free option opens a browser search instead of scraping or using a paid API.
  // Only city and specialty are included; no patient name, MRI, or report is sent.
  const searchQuery = `${specialty} near ${city}`;
  const searchUrl = `https://www.google.com/maps/search/${encodeURIComponent(searchQuery)}`;
  res.json({
    doctors: [],
    search_url: searchUrl,
    search_query: searchQuery,
    disclaimer: 'This opens a live Google Maps search. Results are not verified medical referrals; verify credentials, specialty, availability, and emergency services before contacting a provider.'
  });
});

app.get('/api/scans', requireAuth, async (req, res) => { const scans = await Scan.find(req.query.patient_id ? { patient_id: req.query.patient_id } : {}).sort({ createdAt: -1 }).limit(100); res.json({ scans: scans.map(s => ({ id: s._id.toString(), patient_id: s.patient_id, filename: s.filename, prediction: s.prediction, confidence: s.confidence, probabilities: s.probabilities, metadata: s.metadata, image_preview: s.image_preview, created_at: s.createdAt })) }); });
app.post('/api/scans', requireAuth, async (req, res) => { const p = await upsertPatient(req.body.patient); const s = await Scan.create({ patient_id: p.patient_id, filename: req.body.filename || 'uploaded-scan', prediction: req.body.prediction, confidence: req.body.confidence, probabilities: req.body.probabilities, metadata: req.body.metadata, image_preview: req.body.image_preview || '' }); await audit(req.user, 'create', 'scan', s._id.toString(), `Prediction: ${s.prediction}`); res.json({ id: s._id.toString(), patient_id: s.patient_id, filename: s.filename, prediction: s.prediction, confidence: s.confidence, probabilities: s.probabilities, metadata: s.metadata, image_preview: s.image_preview, created_at: s.createdAt }); });

app.get('/api/reports', requireAuth, async (req, res) => { const reports = await Report.find(req.query.patient_id ? { patient_id: req.query.patient_id } : {}).sort({ createdAt: -1 }).limit(100); res.json({ reports }); });
async function saveReport(req) { const p = await upsertPatient(req.body.patient); const report = await Report.create({ patient_id: p.patient_id, scan_id: req.body.scan_id, doctor_name: req.body.doctor_name || 'Unknown doctor', decision: req.body.decision || 'concur', notes: req.body.notes || '' }); await audit(req.user, 'create', 'report', report._id.toString(), `Decision: ${report.decision}`); return report; }
app.post('/api/reports', requireAuth, requireRole('doctor', 'admin'), async (req, res) => res.json(await saveReport(req)));
app.post('/api/reports/email', requireAuth, requireRole('doctor', 'admin'), async (req, res) => { const report = await saveReport(req); let email; try { email = await sendEmail(req.user.email, 'TumorGANet MRI analysis report', `TumorGANet preliminary MRI analysis report\n\nPatient: ${req.body.patient?.name || 'Unknown'}\nPatient ID: ${req.body.patient?.id || 'Unknown'}\nDoctor: ${req.body.doctor_name || 'Unknown'}\nDecision: ${req.body.decision || 'pending_review'}\n\nNotes: ${req.body.notes || ''}\n\nThis is an AI decision-support report and requires clinician review.`); } catch (error) { email = { status: 'failed', message: `Email delivery failed: ${error.message}` }; } res.json({ report, email }); });
app.get('/api/audit', requireAuth, requireRole('admin'), async (req, res) => { const events = await AuditEvent.find().sort({ createdAt: -1 }).limit(100); res.json({ events }); });

async function proxyAI(req, res, endpoint) {
  try {
    const form = new FormData();
    form.append('file', new Blob([req.file.buffer], { type: req.file.mimetype || 'application/octet-stream' }), req.file.originalname || 'mri-upload');
    const response = await fetch(`${AI_SERVICE_URL}${endpoint}`, { method: 'POST', body: form });
    const body = await response.text();
    res.status(response.status).type(response.headers.get('content-type') || 'application/json').send(body);
  } catch (error) { res.status(502).json({ detail: `AI service unavailable: ${error.message}` }); }
}
app.post('/api/predict', upload.single('file'), (req, res) => req.file ? proxyAI(req, res, '/api/predict') : res.status(400).json({ detail: 'MRI file is required' }));
app.post('/api/predict-dicom', upload.single('file'), (req, res) => req.file ? proxyAI(req, res, '/api/predict-dicom') : res.status(400).json({ detail: 'DICOM file is required' }));

mongoose.connect(process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/tumorganet').then(async () => {
  if (!(await User.exists({ username: 'admin' }))) await User.create({ username: 'admin', email: process.env.TUMORGANET_ADMIN_EMAIL || 'admin@example.com', display_name: 'Administrator', password_hash: await bcrypt.hash(process.env.TUMORGANET_ADMIN_PASSWORD || 'change-me', 12), role: 'admin' });
  app.listen(PORT, () => console.log(`TumorGANet Express API running on http://127.0.0.1:${PORT}`));
}).catch(error => { console.error('MongoDB connection failed:', error.message); process.exit(1); });
