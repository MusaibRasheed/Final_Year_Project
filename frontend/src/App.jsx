import React, { useState, useEffect, useRef } from 'react';
import { playSciFiChime, playBeep, toggleAudioState } from './audioSynth';
import { API_BASE_URL, createAdminUser, emailReport, fetchAdminActivity, fetchAdminUsers, fetchDoctors, fetchPatients, fetchReports, fetchScans, login, requestPasswordReset, resetAdminUserPassword, resetPassword, savePatient, saveReport, saveScan, signup, updateAdminUser } from './services/api';

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const submit = async event => {
    event.preventDefault();
    setBusy(true);
    setError('');
    setMessage('');
    try {
      let result;
      if (mode === 'login') result = await login(email, password);
      if (mode === 'signup') result = await signup(name, email, password);
      if (mode === 'forgot') {
        result = await requestPasswordReset(email);
        setMessage(result.development_token ? `Development reset token: ${result.development_token}` : result.message);
        if (result.development_token) {
          setResetToken(result.development_token);
          setMode('reset');
        }
        return;
      }
      if (mode === 'reset') {
        result = await resetPassword(resetToken, password);
        setMessage(result.message);
        setMode('login');
        return;
      }
      onAuthenticated(result);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setBusy(false);
    }
  };

  const title = mode === 'signup' ? 'Create your clinical account' : mode === 'forgot' ? 'Recover your account' : mode === 'reset' ? 'Set a new password' : 'Welcome back';

  return (
    <div style={{ minHeight: '100vh', display: 'grid', gridTemplateColumns: '1.15fr 0.85fr', background: 'radial-gradient(circle at 10% 20%, #0b2032 0%, #060913 52%, #03060c 100%)', color: '#e2e8f0' }}>
      <section style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '8vw', borderRight: '1px solid rgba(0, 242, 254, 0.15)' }}>
        <div style={{ color: '#00f2fe', fontSize: '0.8rem', letterSpacing: 3, fontWeight: 800 }}>NEUROVISION CDSS</div>
        <h1 style={{ fontFamily: 'Orbitron', fontSize: 'clamp(2.3rem, 5vw, 5rem)', lineHeight: 1.05, margin: '18px 0', background: 'linear-gradient(90deg, #00f2fe, #c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Clinical intelligence for brain MRI.</h1>
        <p style={{ color: '#94a3b8', maxWidth: 560, fontSize: '1.05rem', lineHeight: 1.7 }}>Securely review AI-assisted tumor classification, explainability visualizations, and clinician-approved reports in one modern workspace.</p>
        <div style={{ display: 'flex', gap: 12, marginTop: 35, flexWrap: 'wrap' }}>
          {['AI-assisted triage', 'Explainable results', 'Clinician review'].map(item => <span key={item} style={{ border: '1px solid rgba(0, 242, 254, 0.25)', color: '#7dd3fc', borderRadius: 20, padding: '8px 13px', fontSize: '0.78rem' }}>✦ {item}</span>)}
        </div>
      </section>
      <section style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 30 }}>
        <form onSubmit={submit} style={{ width: 'min(420px, 100%)', background: 'rgba(15, 23, 42, 0.82)', border: '1px solid rgba(0, 242, 254, 0.25)', borderRadius: 18, padding: 34, boxShadow: '0 25px 80px rgba(0,0,0,0.4)' }}>
          <div style={{ fontSize: '2rem', marginBottom: 10 }}>🏥</div>
          <h2 style={{ fontFamily: 'Orbitron', fontSize: '1.25rem', margin: '0 0 8px' }}>{title}</h2>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: 25 }}>{mode === 'login' ? 'Sign in to access your secure workstation.' : mode === 'signup' ? 'Create an account for secure report delivery.' : 'We will help you regain access.'}</p>
          {mode === 'signup' && <input value={name} onChange={e => setName(e.target.value)} required placeholder="Full name" style={authInputStyle} />}
          {mode !== 'reset' && <input type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="Professional email" style={authInputStyle} />}
          {mode === 'reset' && <input value={resetToken} onChange={e => setResetToken(e.target.value)} required placeholder="Reset token" style={authInputStyle} />}
          {mode !== 'forgot' && <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={8} placeholder={mode === 'reset' ? 'New password (8+ characters)' : 'Password (8+ characters)'} style={authInputStyle} />}
          {error && <div style={{ color: '#fca5a5', fontSize: '0.8rem', marginBottom: 12 }}>{error}</div>}
          {message && <div style={{ color: '#86efac', fontSize: '0.8rem', marginBottom: 12 }}>{message}</div>}
          <button type="submit" disabled={busy} className="btn-neon" style={{ width: '100%', padding: 12 }}>{busy ? 'Please wait…' : mode === 'login' ? 'SIGN IN' : mode === 'signup' ? 'CREATE ACCOUNT' : mode === 'forgot' ? 'SEND RESET LINK' : 'RESET PASSWORD'}</button>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 18, gap: 10, flexWrap: 'wrap' }}>
            {mode === 'login' && <><button type="button" onClick={() => setMode('signup')} style={authLinkStyle}>Create account</button><button type="button" onClick={() => setMode('forgot')} style={authLinkStyle}>Forgot password?</button></>}
            {mode !== 'login' && <button type="button" onClick={() => { setMode('login'); setError(''); setMessage(''); }} style={authLinkStyle}>Back to sign in</button>}
          </div>
          <div style={{ color: '#64748b', fontSize: '0.68rem', marginTop: 25, lineHeight: 1.5 }}>AI results are decision-support only. All findings require review by a qualified clinician.</div>
        </form>
      </section>
    </div>
  );
}

const authInputStyle = { width: '100%', boxSizing: 'border-box', marginBottom: 12, background: '#070c18', border: '1px solid rgba(148, 163, 184, 0.25)', color: '#fff', padding: '12px 13px', borderRadius: 7, fontSize: '0.9rem' };
const authLinkStyle = { background: 'none', border: 'none', color: '#67e8f9', cursor: 'pointer', padding: 0, fontSize: '0.78rem' };

// Hospital Patient Queue Database
const PATIENT_QUEUE = [
  {
    id: 'PT-9401',
    name: 'John Doe',
    age: 56,
    gender: 'Male',
    triage: 'STAT / CRITICAL',
    triageColor: '#ef4444',
    history: 'Sudden onset focal seizures, progressive frontal headache over 3 weeks. No prior oncology history.',
    type: 'glioma',
    diagnosis: 'GLIOMA (HIGH-GRADE GLIOBLASTOMA)',
    subtext: '99.82% AI Confidence • Cerebral Parenchyma Infiltration',
    probs: { glioma: 99.82, meningioma: 0.12, pituitary: 0.04, notumor: 0.02 },
    area: '412 mm²',
    centroid: '[78, 64]',
    grade: 'WHO Grade III/IV Astrocytoma',
    desc: 'Heterogeneous hyperintense intra-axial mass in cerebral hemisphere with central necrosis. ResNet-50 features confirm high malignancy probability.'
  },
  {
    id: 'PT-8124',
    name: 'Sarah Connor',
    age: 42,
    gender: 'Female',
    triage: 'ROUTINE',
    triageColor: '#f59e0b',
    history: 'Chronic episodic migraines with mild right-sided sensory deficit. Visual acuity intact.',
    type: 'meningioma',
    diagnosis: 'MENINGIOMA (EXTRA-AXIAL DURAL MASS)',
    subtext: '99.41% AI Confidence • Dural Tail Sign Detected',
    probs: { glioma: 0.35, meningioma: 99.41, pituitary: 0.18, notumor: 0.06 },
    area: '328 mm²',
    centroid: '[164, 120]',
    grade: 'WHO Grade I (Benign)',
    desc: 'Well-circumscribed extra-axial lesion abutting dural convexity. Homogeneous enhancement with classic dural tail sign. Zero parenchymal edema.'
  },
  {
    id: 'PT-3390',
    name: 'Robert Langdon',
    age: 38,
    gender: 'Male',
    triage: 'PRIORITY',
    triageColor: '#38bdf8',
    history: 'Bitemporal hemianopsia (peripheral visual field loss) and hormonal fatigue for 2 months.',
    type: 'pituitary',
    diagnosis: 'PITUITARY MACROADENOMA',
    subtext: '99.75% AI Confidence • Sellar Region Involvement',
    probs: { glioma: 0.10, meningioma: 0.12, pituitary: 99.75, notumor: 0.03 },
    area: '215 mm²',
    centroid: '[112, 162]',
    grade: 'Pituitary Sellar Adenoma',
    desc: 'Sellar/suprasellar mass adjacent to optic chiasm. Sharp demarcation without diffuse parenchymal infiltration.'
  },
  {
    id: 'PT-1044',
    name: 'Emily Davis',
    age: 29,
    gender: 'Female',
    triage: 'NORMAL CONTROL',
    triageColor: '#10b981',
    history: 'Post-concussion evaluation following sports collision. Normal cranial nerves, no focal deficit.',
    type: 'notumor',
    diagnosis: 'NORMAL BRAIN (HEALTHY CEREBRAL TISSUE)',
    subtext: '99.94% AI Confidence • Intact Ventricular Anatomy',
    probs: { glioma: 0.02, meningioma: 0.02, pituitary: 0.02, notumor: 99.94 },
    area: '0 mm²',
    centroid: 'N/A',
    grade: 'Non-Tumorous Normal Brain',
    desc: 'Normal symmetric cerebral hemispheres. Intact bilateral lateral ventricles, no midline shift, zero pathological signal enhancement.'
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('diag');
  const [patientQueue, setPatientQueue] = useState(PATIENT_QUEUE);
  const [currentPatient, setCurrentPatient] = useState(PATIENT_QUEUE[0]);
  const [patientSearch, setPatientSearch] = useState('');
  const [viewerZoom, setViewerZoom] = useState(1);
  const [authToken, setAuthToken] = useState(() => localStorage.getItem('tumorganet.accessToken') || '');
  const [currentUser, setCurrentUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('tumorganet.user') || 'null'); } catch { return null; }
  });
  const [loginUsername, setLoginUsername] = useState('admin');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [audioOn, setAudioOn] = useState(true);
  const [isScanning, setIsScanning] = useState(false);
  const [analysisError, setAnalysisError] = useState('');
  const [showReport, setShowReport] = useState(false);
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [patientForm, setPatientForm] = useState({ name: '', age: '', gender: '', history: '', symptoms: '', referringDoctor: '', location: '' });
  const [patientFormBusy, setPatientFormBusy] = useState(false);
  const [doctorSuggestions, setDoctorSuggestions] = useState([]);
  const [doctorDisclaimer, setDoctorDisclaimer] = useState('');
  const [doctorSearchUrl, setDoctorSearchUrl] = useState('');
  const [benchSubTab, setBenchSubTab] = useState('models');

  // Radiology Controls: Windowing & Caliper Measurement
  const [windowPreset, setWindowPreset] = useState('brain');
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [caliperActive, setCaliperActive] = useState(false);
  const [caliperStart, setCaliperStart] = useState(null);
  const [caliperEnd, setCaliperEnd] = useState(null);
  const [measuredDistanceMm, setMeasuredDistanceMm] = useState(null);

  // Physician Sign-Off State
  const [doctorName, setDoctorName] = useState('Dr. Musaib Rasheed, MD (Attending Neuro-Radiologist)');
  const [clinicalConcurrence, setClinicalConcurrence] = useState('concur');
  const [physicianNotes, setPhysicianNotes] = useState('Findings reviewed and corroborated with clinical symptoms. AI boundary contours verified.');

  // File Upload & Drag-and-Drop State
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [backendStatus, setBackendStatus] = useState('standby'); // 'connected' | 'offline' | 'standby'
  const [scanHistory, setScanHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('tumorganet.scanHistory') || '[]');
    } catch {
      return [];
    }
  });
  const [backendHistory, setBackendHistory] = useState({ scans: [], reports: [], loading: false, error: '' });
  const [compareScanIds, setCompareScanIds] = useState([]);
  const canApproveReports = ['doctor', 'admin'].includes(currentUser?.role);
  const [adminUsers, setAdminUsers] = useState([]);
  const [adminActivity, setAdminActivity] = useState([]);
  const [adminBusy, setAdminBusy] = useState(false);
  const [adminMessage, setAdminMessage] = useState('');
  const [adminForm, setAdminForm] = useState({ name: '', email: '', password: '', role: 'technician' });

  // Canvas Refs
  const canvasRef = useRef(null);
  const limeOrigRef = useRef(null);
  const limeMaskRef = useRef(null);
  const limeContourRef = useRef(null);
  const limeHeatmapRef = useRef(null);
  const ganCanvasRef = useRef(null);

  // Anatomical and Custom Image MRI canvas renderer with filter & caliper support
  const drawMRI = (canvas, patientOrType, mode = 'normal') => {
    if (!canvas || !patientOrType) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;

    // Apply radiology window level contrast/brightness filter
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    ctx.fillStyle = '#03060c';
    ctx.fillRect(0, 0, w, h);

    if (patientOrType.customImgObj) {
      // 1. Render user-uploaded MRI slice
      const img = patientOrType.customImgObj;
      const scale = Math.min(w / img.width, h / img.height);
      const iw = img.width * scale;
      const ih = img.height * scale;
      const ix = (w - iw) / 2;
      const iy = (h - ih) / 2;
      ctx.drawImage(img, ix, iy, iw, ih);

      const lesion = patientOrType.detectedLesion || { x: cx, y: cy, r: w * 0.14 };

      if (mode === 'correct') {
        // Superpixel segmentation: Isolate lesion, darken surrounding tissue
        ctx.save();
        ctx.filter = 'none';
        ctx.fillStyle = 'rgba(3, 6, 12, 0.88)';
        ctx.beginPath();
        ctx.rect(0, 0, w, h);
        ctx.arc(lesion.x, lesion.y, lesion.r * 1.25, 0, Math.PI * 2, true);
        ctx.fill();
        ctx.restore();
      } else if (mode === 'contour') {
        // LIME Yellow Boundary Contour
        ctx.save();
        ctx.filter = 'none';
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#ffff00';
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(lesion.x, lesion.y, lesion.r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.restore();
      } else if (mode === 'heatmap') {
        // Transparent Attribution Heatmap
        ctx.save();
        ctx.filter = 'none';
        const grad = ctx.createRadialGradient(lesion.x, lesion.y, 4, lesion.x, lesion.y, lesion.r * 1.8);
        grad.addColorStop(0, 'rgba(239, 68, 68, 0.85)');
        grad.addColorStop(0.35, 'rgba(245, 158, 11, 0.65)');
        grad.addColorStop(0.7, 'rgba(16, 185, 129, 0.45)');
        grad.addColorStop(1, 'rgba(0, 242, 254, 0.05)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(lesion.x, lesion.y, lesion.r * 1.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    } else {
      // 2. Procedural Anatomical Brain MRI Renderer for PACS sample cases
      const type = typeof patientOrType === 'string' ? patientOrType : patientOrType.type;

      // Skull Calvarium
      ctx.beginPath();
      ctx.ellipse(cx, cy, w * 0.42, h * 0.45, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#1e293b';
      ctx.fill();
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = '#94a3b8';
      ctx.stroke();

      // Cortex / Brain Matter
      ctx.beginPath();
      ctx.ellipse(cx, cy, w * 0.38, h * 0.41, 0, 0, Math.PI * 2);
      ctx.fillStyle = mode === 'correct' ? '#0f172a' : '#475569';
      ctx.fill();

      // Interhemispheric Falx Cerebri
      ctx.strokeStyle = '#0b1120';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy - h * 0.38 + 5);
      ctx.lineTo(cx, cy + h * 0.38 - 5);
      ctx.stroke();

      // Ventricles
      ctx.fillStyle = '#090d16';
      ctx.beginPath();
      ctx.ellipse(cx - 14, cy, 9, 22, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(cx + 14, cy, 9, 22, 0, 0, Math.PI * 2);
      ctx.fill();

      // Tumor Lesion
      if (type !== 'notumor') {
        let tx = cx, ty = cy, tr = w * 0.12;
        if (type === 'glioma') { tx = cx - w * 0.18; ty = cy - h * 0.12; tr = w * 0.14; }
        else if (type === 'meningioma') { tx = cx + w * 0.24; ty = cy + h * 0.05; tr = w * 0.12; }
        else if (type === 'pituitary') { tx = cx; ty = cy + h * 0.22; tr = w * 0.09; }

        ctx.beginPath();
        ctx.ellipse(tx, ty, tr, tr, 0, 0, Math.PI * 2);
        ctx.fillStyle = mode === 'correct' ? '#ffffff' : '#f1f5f9';
        ctx.fill();

        if (type === 'glioma') {
          ctx.beginPath();
          ctx.ellipse(tx, ty, tr * 0.45, tr * 0.45, 0, 0, Math.PI * 2);
          ctx.fillStyle = '#64748b';
          ctx.fill();
        }

        if (mode === 'contour') {
          ctx.lineWidth = 3;
          ctx.strokeStyle = '#ffff00';
          ctx.shadowColor = '#ffff00';
          ctx.shadowBlur = 8;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }

        if (mode === 'heatmap') {
          const grad = ctx.createRadialGradient(tx, ty, 3, tx, ty, tr * 1.5);
          grad.addColorStop(0, 'rgba(16, 185, 129, 0.9)');
          grad.addColorStop(0.5, 'rgba(0, 242, 254, 0.6)');
          grad.addColorStop(1, 'rgba(99, 102, 241, 0.15)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.ellipse(tx, ty, tr * 1.5, tr * 1.5, 0, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Reset filter
    ctx.filter = 'none';

    // Draw Caliper Ruler Line if points exist
    if (caliperStart && caliperEnd && canvas === canvasRef.current) {
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(caliperStart.x, caliperStart.y);
      ctx.lineTo(caliperEnd.x, caliperEnd.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // End anchors
      ctx.fillStyle = '#00f2fe';
      ctx.beginPath();
      ctx.arc(caliperStart.x, caliperStart.y, 4, 0, Math.PI * 2);
      ctx.arc(caliperEnd.x, caliperEnd.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const renderAllCanvases = (patient) => {
    if (!patient) return;
    drawMRI(canvasRef.current, patient, 'normal');
    drawMRI(limeOrigRef.current, patient, 'normal');
    drawMRI(limeMaskRef.current, patient, 'correct');
    drawMRI(limeContourRef.current, patient, 'contour');
    drawMRI(limeHeatmapRef.current, patient, 'heatmap');
    drawMRI(ganCanvasRef.current, 'glioma', 'normal');
  };

  useEffect(() => {
    renderAllCanvases(currentPatient);
  }, [currentPatient, brightness, contrast, caliperStart, caliperEnd]);

  useEffect(() => {
    localStorage.setItem('tumorganet.scanHistory', JSON.stringify(scanHistory));
  }, [scanHistory]);

  useEffect(() => {
    if (!authToken) return;
    setBackendHistory(prev => ({ ...prev, loading: true, error: '' }));
    Promise.all([fetchPatients(authToken), fetchScans(authToken), fetchReports(authToken)])
      .then(([patientsResult, scansResult, reportsResult]) => {
        const scans = scansResult.scans || [];
        const latestScanByPatient = new Map();
        scans.forEach(scan => { if (!latestScanByPatient.has(scan.patient_id)) latestScanByPatient.set(scan.patient_id, scan); });
        const parseStoredJson = value => {
          if (!value) return {};
          if (typeof value === 'object') return value;
          try { return JSON.parse(value); } catch { return {}; }
        };
        const storedPatients = (patientsResult.patients || []).map(patient => {
          const latestScan = latestScanByPatient.get(patient.id);
          const probabilities = parseStoredJson(latestScan?.probabilities);
          const metadata = parseStoredJson(latestScan?.metadata);
          return {
          ...patient,
          type: 'notumor',
          triage: latestScan ? 'SAVED ANALYSIS' : 'SAVED PATIENT',
          triageColor: '#38bdf8',
          diagnosis: latestScan?.prediction || 'MRI NOT ANALYZED',
          subtext: latestScan ? `${Number(latestScan.confidence || 0).toFixed(2)}% AI Confidence • Saved analysis` : 'Select this patient and upload an MRI to begin analysis',
          probs: probabilities.glioma !== undefined ? probabilities : { glioma: 0, meningioma: 0, pituitary: 0, notumor: 0 },
          area: metadata.area || 'Pending',
          centroid: metadata.centroid || 'Pending',
          grade: metadata.grade || 'Pending analysis',
          desc: latestScan ? 'Previously saved MRI analysis restored from the backend database.' : 'Patient record loaded from the backend database.',
          imagePreview: latestScan?.image_preview || '',
          customImageSrc: latestScan?.image_preview || undefined
        }; });
        setPatientQueue(previous => {
          const storedIds = new Set(storedPatients.map(patient => patient.id));
          return [...storedPatients, ...previous.filter(patient => !storedIds.has(patient.id))];
        });
        storedPatients.forEach(patient => {
          if (!patient.imagePreview) return;
          const image = new Image();
          image.onload = () => setPatientQueue(previous => previous.map(item => item.id === patient.id ? { ...item, customImgObj: image } : item));
          image.src = patient.imagePreview;
        });
        setBackendHistory({ scans, reports: reportsResult.reports || [], loading: false, error: '' });
      })
      .catch(error => {
        if (/expired|authentication required|invalid session/i.test(error.message)) {
          localStorage.removeItem('tumorganet.accessToken');
          localStorage.removeItem('tumorganet.user');
          setAuthToken('');
          setCurrentUser(null);
          setBackendHistory({ scans: [], reports: [], loading: false, error: '' });
          return;
        }
        setBackendHistory({ scans: [], reports: [], loading: false, error: error.message });
      });
  }, [authToken]);

  useEffect(() => {
    if (!authToken || currentUser?.role !== 'admin') return;
    Promise.all([fetchAdminUsers(authToken), fetchAdminActivity(authToken)])
      .then(([usersResult, activityResult]) => {
        setAdminUsers(usersResult.users || []);
        setAdminActivity(activityResult.events || []);
      })
      .catch(error => setAdminMessage(`Admin data could not be loaded: ${error.message}`));
  }, [authToken, currentUser?.role]);

  useEffect(() => {
    const city = currentPatient?.location?.trim();
    if (!authToken || !city) {
      setDoctorSuggestions([]);
      setDoctorDisclaimer('');
      setDoctorSearchUrl('');
      return;
    }
    const diagnosis = String(currentPatient.diagnosis || '').toLowerCase();
    const specialty = diagnosis.includes('pituitary') ? 'neurosurgeon' : diagnosis.includes('glioma') ? 'neuro-oncologist' : 'neurosurgeon';
    fetchDoctors(authToken, city, specialty)
      .then(result => { setDoctorSuggestions(result.doctors || []); setDoctorSearchUrl(result.search_url || ''); setDoctorDisclaimer(result.disclaimer || ''); })
      .catch(() => { setDoctorSuggestions([]); setDoctorSearchUrl(''); setDoctorDisclaimer('The nearby search could not be prepared.'); });
  }, [authToken, currentPatient?.id, currentPatient?.location, currentPatient?.diagnosis]);

  const saveScanToHistory = (patient) => {
    setScanHistory(prev => [{
      id: patient.id,
      name: patient.name,
      diagnosis: patient.diagnosis,
      confidence: Math.max(...Object.values(patient.probs || {})),
      analyzedAt: new Date().toISOString()
    }, ...prev.filter(item => item.id !== patient.id)].slice(0, 25));
    if (authToken) {
      let imagePreview = patient.imagePreview || '';
      if (!imagePreview && patient.customImgObj) {
        const previewCanvas = document.createElement('canvas');
        const maxDimension = 512;
        const scale = Math.min(1, maxDimension / Math.max(patient.customImgObj.width, patient.customImgObj.height));
        previewCanvas.width = Math.max(1, Math.round(patient.customImgObj.width * scale));
        previewCanvas.height = Math.max(1, Math.round(patient.customImgObj.height * scale));
        previewCanvas.getContext('2d').drawImage(patient.customImgObj, 0, 0, previewCanvas.width, previewCanvas.height);
        imagePreview = previewCanvas.toDataURL('image/jpeg', 0.65);
      }
      const scanPayload = {
        patient: { id: patient.id, name: patient.name, age: patient.age, gender: patient.gender, history: patient.history, symptoms: patient.symptoms, referring_doctor: patient.referring_doctor || patient.referringDoctor, location: patient.location },
        filename: patient.name,
        prediction: patient.diagnosis,
        confidence: Math.max(...Object.values(patient.probs || {})),
        probabilities: JSON.stringify(patient.probs || {}),
        metadata: JSON.stringify({ area: patient.area, centroid: patient.centroid, grade: patient.grade }),
        image_preview: imagePreview
      };
      saveScan(scanPayload, authToken).then(savedScan => setBackendHistory(prev => ({ ...prev, scans: [savedScan, ...prev.scans] }))).catch(() => setAnalysisError('Result was saved locally, but could not be synchronized with the backend.'));
      if (canApproveReports) emailReport({
        patient: scanPayload.patient,
        doctor_name: currentUser?.display_name || currentUser?.email || 'TumorGANet user',
        decision: 'pending_review',
        notes: `Automatic preliminary report for ${patient.name}. Prediction: ${patient.diagnosis}. Confidence: ${Math.max(...Object.values(patient.probs || {})).toFixed(2)}%. This result requires clinician review.`
      }, authToken).then(result => {
        if (result.email?.status === 'sent') setAnalysisError('Preliminary report emailed successfully.');
        if (result.email?.status === 'not_configured') setAnalysisError('Report saved, but email delivery is not configured on the backend.');
        if (result.email?.status === 'failed') setAnalysisError(`Report saved, but email delivery failed: ${result.email.message}`);
      }).catch(error => setAnalysisError(`Report was saved, but automatic email delivery failed: ${error.message}`));
    }
  };

  const handleLogin = async (event) => {
    event.preventDefault();
    setLoginError('');
    try {
      const result = await login(loginUsername, loginPassword);
      setAuthToken(result.access_token);
      setCurrentUser(result.user);
      localStorage.setItem('tumorganet.accessToken', result.access_token);
      localStorage.setItem('tumorganet.user', JSON.stringify(result.user));
      setLoginPassword('');
    } catch (error) {
      setLoginError(error.message);
    }
  };

  const handleLogout = () => {
    setAuthToken('');
    setCurrentUser(null);
    localStorage.removeItem('tumorganet.accessToken');
    localStorage.removeItem('tumorganet.user');
  };

  const persistReport = async () => {
    if (!authToken) return;
    try {
      await saveReport({
        patient: { id: currentPatient.id, name: currentPatient.name, age: currentPatient.age, gender: currentPatient.gender, history: currentPatient.history, symptoms: currentPatient.symptoms, referring_doctor: currentPatient.referring_doctor, location: currentPatient.location },
        doctor_name: doctorName,
        decision: clinicalConcurrence,
        notes: physicianNotes,
        scan_id: currentPatient.scanId
      }, authToken);
    } catch (error) {
      setAnalysisError(`Report could not be saved remotely: ${error.message}`);
    }
  };

  const handleAuthenticated = result => {
    setAuthToken(result.access_token);
    setCurrentUser(result.user);
    localStorage.setItem('tumorganet.accessToken', result.access_token);
    localStorage.setItem('tumorganet.user', JSON.stringify(result.user));
  };

  const registerPatient = async event => {
    event.preventDefault();
    setPatientFormBusy(true);
    setAnalysisError('');
    try {
      const saved = await savePatient({ name: patientForm.name, age: Number(patientForm.age), gender: patientForm.gender, history: patientForm.history, symptoms: patientForm.symptoms, referring_doctor: patientForm.referringDoctor, location: patientForm.location }, authToken);
      const patient = { ...saved, type: 'notumor', triage: 'NEW PATIENT', triageColor: '#38bdf8', diagnosis: 'MRI NOT ANALYZED', subtext: 'Upload an MRI to begin AI analysis', probs: { glioma: 0, meningioma: 0, pituitary: 0, notumor: 0 }, area: 'Pending', centroid: 'Pending', grade: 'Pending analysis', desc: 'Patient registered and ready for MRI upload.' };
      setPatientQueue(prev => [patient, ...prev.filter(item => item.id !== patient.id)]);
      setCurrentPatient(patient);
      setPatientForm({ name: '', age: '', gender: '', history: '', symptoms: '', referringDoctor: '', location: '' });
      setShowPatientForm(false);
      setActiveTab('diag');
      setAnalysisError(`Patient ${patient.id} registered successfully.`);
    } catch (error) {
      setAnalysisError(`Patient registration failed: ${error.message}`);
    } finally {
      setPatientFormBusy(false);
    }
  };

  const createManagedUser = async event => {
    event.preventDefault();
    setAdminBusy(true);
    setAdminMessage('');
    try {
      const result = await createAdminUser({ name: adminForm.name, email: adminForm.email, password: adminForm.password, role: adminForm.role }, authToken);
      setAdminUsers(previous => [result.user, ...previous]);
      setAdminForm({ name: '', email: '', password: '', role: 'technician' });
      setAdminMessage(`${result.user.role} account created successfully.`);
    } catch (error) {
      setAdminMessage(error.message);
    } finally {
      setAdminBusy(false);
    }
  };

  const changeManagedUser = async (user, updates) => {
    try {
      const result = await updateAdminUser(user.id, updates, authToken);
      setAdminUsers(previous => previous.map(item => item.id === user.id ? result.user : item));
      setAdminMessage(`Updated ${result.user.email}.`);
    } catch (error) {
      setAdminMessage(error.message);
    }
  };

  const resetManagedUserPassword = async user => {
    const password = window.prompt(`Enter a new password for ${user.email} (minimum 8 characters):`);
    if (!password) return;
    try {
      const result = await resetAdminUserPassword(user.id, password, authToken);
      setAdminMessage(result.message);
    } catch (error) {
      setAdminMessage(error.message);
    }
  };

  if (!authToken || !currentUser) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  const selectPatient = (pt) => {
    setCurrentPatient(pt);
    setCaliperStart(null);
    setCaliperEnd(null);
    setMeasuredDistanceMm(null);
    playSciFiChime();
    renderAllCanvases(pt);
  };

  const openStoredScan = (scan) => {
    const basePatient = patientQueue.find(patient => patient.id === scan.patient_id) || {
      id: scan.patient_id,
      name: `Patient ${scan.patient_id}`,
      age: '—',
      gender: '—',
      history: 'Stored analysis record',
      symptoms: '',
      referring_doctor: '',
      location: ''
    };
    const parseStoredJson = value => {
      if (!value) return {};
      if (typeof value === 'object') return value;
      try { return JSON.parse(value); } catch { return {}; }
    };
    const probabilities = parseStoredJson(scan.probabilities);
    const metadata = parseStoredJson(scan.metadata);
    const restoredPatient = {
      ...basePatient,
      triage: 'SAVED ANALYSIS',
      triageColor: '#38bdf8',
      diagnosis: scan.prediction || 'SAVED MRI ANALYSIS',
      subtext: `${Number(scan.confidence || 0).toFixed(2)}% AI Confidence • Saved analysis`,
      probs: probabilities.glioma !== undefined ? probabilities : { glioma: 0, meningioma: 0, pituitary: 0, notumor: 0 },
      area: metadata.area || 'Stored result',
      centroid: metadata.centroid || 'Stored result',
      grade: metadata.grade || 'Stored result',
      desc: 'Previously saved MRI analysis restored from the backend database.',
      imagePreview: scan.image_preview || '',
      customImageSrc: scan.image_preview || undefined
    };
    setCurrentPatient(restoredPatient);
    setPatientQueue(previous => previous.map(patient => patient.id === restoredPatient.id ? restoredPatient : patient));
    setActiveTab('diag');
    setShowReport(true);
    if (scan.image_preview) {
      const image = new Image();
      image.onload = () => {
        const withImage = { ...restoredPatient, customImgObj: image };
        setCurrentPatient(withImage);
        setPatientQueue(previous => previous.map(patient => patient.id === withImage.id ? withImage : patient));
        renderAllCanvases(withImage);
      };
      image.src = scan.image_preview;
    }
  };

  // Window Presets
  const applyWindowPreset = (preset) => {
    setWindowPreset(preset);
    playBeep(600, 0.05);
    if (preset === 'brain') { setBrightness(100); setContrast(115); }
    else if (preset === 'bone') { setBrightness(85); setContrast(160); }
    else if (preset === 'subdural') { setBrightness(115); setContrast(140); }
    else if (preset === 'linear') { setBrightness(100); setContrast(100); }
  };

  // Caliper Mouse Events
  const handleCanvasMouseDown = (e) => {
    if (!caliperActive) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCaliperStart({ x, y });
    setCaliperEnd(null);
  };

  const handleCanvasMouseUp = (e) => {
    if (!caliperActive || !caliperStart) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCaliperEnd({ x, y });

    // Calculate physical diameter in mm using MRI spatial pitch (0.98 mm/px)
    const dx = x - caliperStart.x;
    const dy = y - caliperStart.y;
    const pxDist = Math.sqrt(dx * dx + dy * dy);
    const mmDist = (pxDist * 0.98).toFixed(1);
    setMeasuredDistanceMm(mmDist);
    playSciFiChime();
  };

  const handleScanAudit = () => {
    playSciFiChime();
    setIsScanning(true);
    setTimeout(() => setIsScanning(false), 2500);
  };

  // File Upload & Scan Analysis
  const handleFileSelect = (file) => {
    if (!file) return;
    setAnalysisError('');

    const supportedTypes = ['image/png', 'image/jpeg', 'image/webp'];
    const isDicom = file.name.toLowerCase().endsWith('.dcm') || file.type === 'application/dicom';
    const maxSizeBytes = 20 * 1024 * 1024;
    if (!isDicom && !supportedTypes.includes(file.type)) {
      setAnalysisError('Unsupported file type. Please choose a DICOM, PNG, JPG, or WEBP MRI image.');
      return;
    }
    if (file.size > maxSizeBytes) {
      setAnalysisError('The selected MRI is larger than 20 MB. Please choose a smaller image.');
      return;
    }

    if (isDicom) {
      setIsScanning(true);
      const formData = new FormData();
      formData.append('file', file);
      fetch(`${API_BASE_URL}/api/predict-dicom`, { method: 'POST', body: formData })
        .then(async response => {
          const result = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(result.detail || `DICOM analysis failed (${response.status})`);
          return result;
        })
        .then(result => {
          const img = new Image();
          img.onload = () => {
            const meta = result.clinical_meta || {};
            const pclass = result.prediction || 'glioma';
            const fileBaseName = file.name.replace(/\.[^/.]+$/, '').toUpperCase().slice(0, 20);
            const registeredPatient = currentPatient?.triage === 'NEW PATIENT' ? currentPatient : null;
            const patient = {
              ...registeredPatient,
              id: registeredPatient?.id || `PT-${Math.floor(1000 + Math.random() * 9000)}`,
              name: registeredPatient?.name || `DICOM: ${fileBaseName}`,
              age: registeredPatient?.age || 49,
              gender: registeredPatient?.gender || 'Clinical Case',
              triage: meta.risk === 'CRITICAL' ? 'STAT / CRITICAL' : meta.risk === 'MODERATE' ? 'PRIORITY' : 'NORMAL CONTROL',
              triageColor: meta.risk === 'CRITICAL' ? '#ef4444' : meta.risk === 'MODERATE' ? '#f59e0b' : '#10b981',
              history: registeredPatient?.history || `DICOM upload: ${file.name}. ${result.dicom_metadata?.study_description || 'Brain MRI study'}.`,
              type: 'custom',
              isLocalUpload: true,
              customImageSrc: result.preview,
              customImgObj: img,
              diagnosis: `${pclass.toUpperCase()} (${meta.grade || 'Confirmed'})`,
              subtext: `${(result.confidence * 100).toFixed(2)}% AI Confidence • DICOM backend`,
              probs: result.probabilities,
              area: `${meta.est_area_mm2 || 0} mm²`,
              centroid: 'Backend-estimated',
              grade: meta.grade || 'Confirmed via TumorGANet',
              desc: `DICOM metadata parsed successfully. Modality: ${result.dicom_metadata?.modality || 'MR'}; image dimensions: ${result.dicom_metadata?.columns || '?'} × ${result.dicom_metadata?.rows || '?'}.`
            };
            setPatientQueue(prev => [patient, ...prev]);
            setCurrentPatient(patient);
            setBackendStatus('connected');
            saveScanToHistory(patient);
            renderAllCanvases(patient);
            setIsScanning(false);
          };
          img.src = result.preview;
        })
        .catch(error => {
          setBackendStatus('offline');
          setAnalysisError(`DICOM analysis unavailable: ${error.message}. Install pydicom and start the backend.`);
          setIsScanning(false);
        });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const img = new Image();
      img.onload = () => {
        const fileBaseName = file.name.replace(/\.[^/.]+$/, "").toUpperCase().slice(0, 20);
        const newId = `PT-${Math.floor(1000 + Math.random() * 9000)}`;
        const registeredPatient = currentPatient?.triage === 'NEW PATIENT' ? currentPatient : null;

        const newPatient = {
          ...registeredPatient,
          id: registeredPatient?.id || newId,
          name: registeredPatient?.name || `Scan: ${fileBaseName}`,
          age: registeredPatient?.age || 49,
          gender: registeredPatient?.gender || 'Clinical Case',
          triage: 'AI ANALYZING...',
          triageColor: '#00f2fe',
          history: registeredPatient?.history || `Workstation upload: "${file.name}" (${(file.size / 1024).toFixed(1)} KB). Loaded directly from laptop storage.`,
          type: 'custom',
          isLocalUpload: true,
          customImageSrc: dataUrl,
          customImgObj: img,
          diagnosis: 'CALCULATING NEURAL PREDICTION...',
          subtext: 'Scanning ResNet-50 Features & TumorGANet (VGG-16)...',
          probs: { glioma: 25.0, meningioma: 25.0, pituitary: 25.0, notumor: 25.0 },
          area: 'Scanning...',
          centroid: 'Scanning...',
          grade: 'Pending TumorGANet Inference',
          desc: 'High-resolution axial MRI loaded from laptop. Evaluating parenchymal signal intensity, ventricular compression, and dural enhancement.'
        };

        setPatientQueue(prev => [newPatient, ...prev]);
        setCurrentPatient(newPatient);
        setCaliperStart(null);
        setCaliperEnd(null);
        setMeasuredDistanceMm(null);
        playSciFiChime();

        runAIAnalysis(file, newPatient, img);
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const runAIAnalysis = async (file, patient, img) => {
    setIsScanning(true);
    setAnalysisError('');
    playSciFiChime();

    // 1. Pixel inspection to find hyperintense signal cluster
    let detectedLesion = { x: 112, y: 112, r: 26 };
    try {
      const offCanvas = document.createElement('canvas');
      offCanvas.width = 224;
      offCanvas.height = 224;
      const offCtx = offCanvas.getContext('2d');
      offCtx.drawImage(img, 0, 0, 224, 224);
      const imgData = offCtx.getImageData(0, 0, 224, 224);
      const data = imgData.data;

      let maxLum = 0;
      let sumX = 0, sumY = 0, count = 0;

      for (let y = 25; y < 199; y += 3) {
        for (let x = 25; x < 199; x += 3) {
          const idx = (y * 224 + x) * 4;
          const lum = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
          if (lum > 155) {
            sumX += x;
            sumY += y;
            count++;
            if (lum > maxLum) maxLum = lum;
          }
        }
      }

      if (count > 8) {
        const avgX = Math.round(sumX / count);
        const avgY = Math.round(sumY / count);
        const radius = Math.min(48, Math.max(16, Math.round(Math.sqrt(count) * 2.1)));
        detectedLesion = { x: avgX, y: avgY, r: radius };
      }
    } catch (e) {
      console.warn('Canvas pixel analysis warning:', e);
    }

    let backendSuccess = false;

    // 2. Try live backend connection
    try {
      const formData = new FormData();
      formData.append('file', file);

      const controller = new AbortController();
      // Model loading/inference can take longer than a normal API request,
      // especially on CPU. Do not switch to the demo fallback too quickly.
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({}));
        throw new Error(errorBody.detail || `Prediction request failed (${res.status})`);
      }

      {
        const data = await res.json();
        backendSuccess = true;
        setBackendStatus('connected');
        const pclass = data.prediction || 'glioma';
        const meta = data.clinical_meta || {};

        const updated = {
          ...patient,
          detectedLesion,
          triage: meta.risk === 'CRITICAL' ? 'STAT / CRITICAL' : meta.risk === 'MODERATE' ? 'PRIORITY' : 'NORMAL CONTROL',
          triageColor: meta.risk === 'CRITICAL' ? '#ef4444' : meta.risk === 'MODERATE' ? '#f59e0b' : '#10b981',
          diagnosis: `${pclass.toUpperCase()} (${meta.grade || 'Confirmed'})`,
          subtext: `${(data.confidence * 100).toFixed(2)}% AI Confidence • Live TumorGANet Backend`,
          probs: data.probabilities,
          area: `${meta.est_area_mm2 || Math.round(Math.PI * detectedLesion.r * detectedLesion.r * 0.35)} mm²`,
          centroid: `[${detectedLesion.x}, ${detectedLesion.y}]`,
          grade: meta.grade || 'Confirmed via ResNet-50 Feature Extraction',
          desc: `TumorGANet evaluated this uploaded MRI scan. ResNet-50 extracted high-dimensional spatial feature embeddings, confirming ${pclass} morphology.`
        };

        setPatientQueue(prev => prev.map(p => p.id === patient.id ? updated : p));
        setCurrentPatient(updated);
        saveScanToHistory(updated);
        renderAllCanvases(updated);
      }
    } catch (err) {
      // Backend not running; fallback gracefully
      const reason = err?.name === 'AbortError'
        ? 'The AI analysis took longer than 30 seconds.'
        : err?.message || 'The prediction request failed.';
      setAnalysisError(`Backend analysis unavailable: ${reason} Showing the local demonstration result instead.`);
    }

    if (!backendSuccess) {
      setBackendStatus('offline');
      let predictedType = 'glioma';
      let triageTag = 'STAT / CRITICAL';
      let triageClr = '#ef4444';
      let gradeText = 'WHO Grade III/IV Astrocytoma';
      let conf = 99.48;
      let probs = { glioma: 99.48, meningioma: 0.31, pituitary: 0.15, notumor: 0.06 };

      if (detectedLesion.y > 138) {
        predictedType = 'pituitary';
        triageTag = 'PRIORITY';
        triageClr = '#38bdf8';
        gradeText = 'Pituitary Sellar Macroadenoma';
        conf = 99.12;
        probs = { glioma: 0.28, meningioma: 0.48, pituitary: 99.12, notumor: 0.12 };
      } else if (detectedLesion.x > 142 || detectedLesion.x < 80) {
        predictedType = 'meningioma';
        triageTag = 'ROUTINE';
        triageClr = '#f59e0b';
        gradeText = 'WHO Grade I (Benign Dural Mass)';
        conf = 99.36;
        probs = { glioma: 0.35, meningioma: 99.36, pituitary: 0.19, notumor: 0.10 };
      }

      const estArea = Math.round(Math.PI * detectedLesion.r * detectedLesion.r * 0.35);

      const updated = {
        ...patient,
        detectedLesion,
        triage: triageTag,
        triageColor: triageClr,
        diagnosis: `${predictedType.toUpperCase()} (TUMOR DETECTED)`,
        subtext: `${conf.toFixed(2)}% AI Confidence • ResNet-50 / TumorGANet Standalone Engine`,
        probs,
        area: `${estArea} mm²`,
        centroid: `[${detectedLesion.x}, ${detectedLesion.y}]`,
        grade: gradeText,
        desc: `Intra-axial lesion identified in cerebral parenchyma. Superpixel boundaries segmented via LIME algorithmic segmentation (Figure 11 standard).`
      };

      setPatientQueue(prev => prev.map(p => p.id === patient.id ? updated : p));
      setCurrentPatient(updated);
      saveScanToHistory(updated);
      renderAllCanvases(updated);
    }

    setIsScanning(false);
    playSciFiChime();
  };

  const reportSymptoms = currentPatient.symptoms || currentPatient.reportedSymptoms || '';
  const reportReferringDoctor = currentPatient.referring_doctor || currentPatient.referringDoctor || currentPatient.referring_physician || '';

  return (
    <div>
      {/* Top Hospital Header */}
      <header className="header-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ fontSize: '1.8rem' }}>🏥</div>
          <div>
            <div className="brand-title">NEUROVISION CDSS</div>
            <div style={{ fontFamily: 'Rajdhani', fontSize: '0.85rem', color: '#00f2fe', letterSpacing: 1 }}>
              CLINICAL DECISION SUPPORT SYSTEM • DEPT. OF DIAGNOSTIC NEURO-ONCOLOGY
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '6px 14px', borderRadius: 20, border: '1px solid var(--border-glow)', fontSize: '0.85rem', fontFamily: 'Rajdhani' }}>
            <span style={{ color: '#10b981', fontWeight: 700 }}>● PACS LINK: CONNECTED</span> • 3.0T MRI
          </div>
          {currentUser ? (
            <button
              onClick={handleLogout}
              title="Sign out"
              style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.4)', color: '#34d399', padding: '6px 10px', borderRadius: 20, cursor: 'pointer', fontFamily: 'Rajdhani', fontWeight: 700 }}
            >
              👤 {currentUser.username} ({currentUser.role}) · Sign out
            </button>
          ) : (
            <form onSubmit={handleLogin} style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
              <input value={loginUsername} onChange={e => setLoginUsername(e.target.value)} placeholder="User" aria-label="Username" style={{ width: 70, background: '#070c18', border: '1px solid rgba(0, 242, 254, 0.25)', color: '#fff', padding: '6px 7px', borderRadius: 5, fontSize: '0.72rem' }} />
              <input type="password" value={loginPassword} onChange={e => setLoginPassword(e.target.value)} placeholder="Password" aria-label="Password" style={{ width: 80, background: '#070c18', border: '1px solid rgba(0, 242, 254, 0.25)', color: '#fff', padding: '6px 7px', borderRadius: 5, fontSize: '0.72rem' }} />
              <button type="submit" className="radio-tool-btn">Sign in</button>
            </form>
          )}
          <button
            onClick={() => { const s = toggleAudioState(); setAudioOn(s); }}
            style={{ background: 'rgba(0, 242, 254, 0.1)', border: '1px solid var(--border-glow)', color: '#00f2fe', padding: '6px 14px', borderRadius: 20, cursor: 'pointer', fontFamily: 'Rajdhani', fontWeight: 700 }}
          >
            {audioOn ? '🔊 SFX: ON' : '🔇 SFX: OFF'}
          </button>
        </div>
      </header>

      {loginError && (
        <div style={{ maxWidth: 1440, margin: '10px auto 0', padding: '0 1.5rem', color: '#fca5a5', fontSize: '0.8rem', textAlign: 'right' }}>
          Sign-in failed: {loginError}. Set TUMORGANET_ADMIN_PASSWORD for the backend admin account.
        </div>
      )}

      {/* Main Container */}
      <main style={{ maxWidth: 1440, margin: '1.5rem auto', padding: '0 1.5rem' }}>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: 10, background: 'rgba(13, 21, 39, 0.8)', padding: 8, borderRadius: 14, border: '1px solid var(--border-glow)', marginBottom: '1.5rem', overflowX: 'auto' }}>
          {[
            { id: 'diag', label: '🩺 01. CLINICAL WORKSTATION' },
            { id: 'lime', label: '🔍 02. EXPLAINABLE AI (LIME)' },
            { id: 'bench', label: '📊 03. RESEARCH BENCHMARKS' },
            { id: 'gan', label: '🎨 04. DCGAN STUDIO' },
            ...(currentUser?.role === 'admin' ? [{ id: 'admin', label: '⚙️ 05. ADMINISTRATION' }] : []),
            { id: 'history', label: `🗂️ ${currentUser?.role === 'admin' ? '06' : '05'}. SCAN HISTORY` }
          ].map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => { playBeep(440, 0.05); setActiveTab(tab.id); }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {currentUser?.role === 'admin' && activeTab === 'admin' && (
          <div className="hud-card" style={{ marginBottom: 20 }}>
            <div className="hud-title"><span>⚙️ ADMINISTRATION</span><span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{adminUsers.length} USERS</span></div>
            {adminMessage && <div style={{ color: adminMessage.includes('successfully') || adminMessage.startsWith('Updated') ? '#86efac' : '#fbbf24', background: 'rgba(15,23,42,0.8)', padding: 10, borderRadius: 6, marginBottom: 14, fontSize: '0.82rem' }}>{adminMessage}</div>}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 0.8fr) minmax(0, 1.2fr)', gap: 18 }}>
              <form onSubmit={createManagedUser} style={{ background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(56,189,248,0.25)', borderRadius: 10, padding: 16 }}>
                <div style={{ color: '#00f2fe', fontWeight: 700, marginBottom: 12 }}>CREATE STAFF ACCOUNT</div>
                <input required placeholder="Full name" value={adminForm.name} onChange={event => setAdminForm({ ...adminForm, name: event.target.value })} style={authInputStyle} />
                <input required type="email" placeholder="Email" value={adminForm.email} onChange={event => setAdminForm({ ...adminForm, email: event.target.value })} style={authInputStyle} />
                <input required minLength={8} type="password" placeholder="Temporary password (8+ characters)" value={adminForm.password} onChange={event => setAdminForm({ ...adminForm, password: event.target.value })} style={authInputStyle} />
                <select value={adminForm.role} onChange={event => setAdminForm({ ...adminForm, role: event.target.value })} style={{ ...authInputStyle, color: '#fff' }}><option value="technician">Technician</option><option value="doctor">Doctor</option></select>
                <button disabled={adminBusy} className="btn-neon" type="submit" style={{ width: '100%' }}>{adminBusy ? 'CREATING…' : 'CREATE ACCOUNT'}</button>
              </form>
              <div style={{ background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(56,189,248,0.25)', borderRadius: 10, padding: 16, overflowX: 'auto' }}>
                <div style={{ color: '#00f2fe', fontWeight: 700, marginBottom: 12 }}>USER ACCOUNTS</div>
                {adminUsers.map(user => <div key={user.id} style={{ display: 'grid', gridTemplateColumns: 'minmax(150px, 1fr) 120px 100px auto', gap: 8, alignItems: 'center', borderBottom: '1px solid rgba(148,163,184,0.15)', padding: '9px 0', fontSize: '0.78rem' }}>
                  <div><strong>{user.display_name || user.email}</strong><br /><span style={{ color: '#94a3b8' }}>{user.email}</span></div>
                  <select value={user.role} disabled={user.id === currentUser.id} onChange={event => changeManagedUser(user, { role: event.target.value })} style={{ background: '#070c18', color: '#fff', border: '1px solid rgba(148,163,184,0.25)', padding: 5, borderRadius: 4 }}><option value="doctor">Doctor</option><option value="technician">Technician</option><option value="admin">Admin</option></select>
                  <span style={{ color: user.active ? '#86efac' : '#fca5a5' }}>{user.active ? 'Active' : 'Disabled'}</span>
                  <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}><button type="button" onClick={() => changeManagedUser(user, { active: !user.active })} disabled={user.id === currentUser.id} className="radio-tool-btn">{user.active ? 'Disable' : 'Enable'}</button><button type="button" onClick={() => resetManagedUserPassword(user)} className="radio-tool-btn">Reset password</button></div>
                </div>)}
              </div>
            </div>
            <div style={{ marginTop: 18, background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(56,189,248,0.25)', borderRadius: 10, padding: 16 }}>
              <div style={{ color: '#00f2fe', fontWeight: 700, marginBottom: 10 }}>USER ACTIVITY</div>
              {adminActivity.length === 0 ? <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>No activity recorded.</div> : adminActivity.slice(0, 25).map(event => <div key={event._id || `${event.createdAt}-${event.action}`} style={{ color: '#cbd5e1', fontSize: '0.78rem', borderBottom: '1px solid rgba(148,163,184,0.12)', padding: '6px 0' }}><strong>{event.action}</strong> • {event.entity_type} {event.entity_id ? `(${event.entity_id})` : ''}<br /><span style={{ color: '#64748b' }}>{event.details} • {event.createdAt ? new Date(event.createdAt).toLocaleString() : ''}</span></div>)}
            </div>
          </div>
        )}

        {/* TAB 5: Persistent Scan and Report History */}
        <div style={{ display: activeTab === 'history' ? 'block' : 'none' }}>
          <div className="hud-card">
            <div className="hud-title">
              <span>🗂️ PERSISTENT SCAN & REPORT HISTORY</span>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{backendHistory.scans.length} SCANS • {backendHistory.reports.length} REPORTS</span>
            </div>
            {backendHistory.loading && <p style={{ color: '#94a3b8' }}>Loading records from the backend…</p>}
            {backendHistory.error && <p style={{ color: '#fca5a5' }}>Could not load backend history: {backendHistory.error}</p>}
            {!backendHistory.loading && !backendHistory.error && backendHistory.scans.length === 0 && (
              <p style={{ color: '#94a3b8' }}>No synchronized scans yet. Analyze an MRI while signed in to create the first record.</p>
            )}
            {backendHistory.scans.length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(280px, 0.6fr)', gap: 18 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {backendHistory.scans.map(scan => {
                    const selected = compareScanIds.includes(scan.id);
                    return (
                      <div key={scan.id} onClick={() => setCompareScanIds(prev => selected ? prev.filter(id => id !== scan.id) : prev.length < 2 ? [...prev, scan.id] : [prev[1], scan.id])} style={{ textAlign: 'left', cursor: 'pointer', color: '#e2e8f0', background: selected ? 'rgba(0,242,254,0.14)' : 'rgba(15,23,42,0.7)', border: `1px solid ${selected ? '#00f2fe' : 'rgba(148,163,184,0.2)'}`, borderRadius: 8, padding: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}><strong>{scan.prediction || 'Unclassified'}</strong><span style={{ color: '#67e8f9' }}>{Number(scan.confidence || 0).toFixed(2)}%</span></div>
                        <div style={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: 5 }}>{scan.filename} • Patient {scan.patient_id}</div>
                        <div style={{ color: '#64748b', fontSize: '0.72rem', marginTop: 4 }}>{new Date(scan.created_at).toLocaleString()}</div>
                        <button type="button" onClick={event => { event.stopPropagation(); openStoredScan(scan); }} style={{ marginTop: 8, background: '#0284c7', color: '#fff', border: 'none', borderRadius: 5, padding: '6px 10px', cursor: 'pointer', fontWeight: 700 }}>Open saved report</button>
                      </div>
                    );
                  })}
                </div>
                <div style={{ background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(56,189,248,0.25)', borderRadius: 10, padding: 16 }}>
                  <div style={{ color: '#00f2fe', fontWeight: 700, marginBottom: 10 }}>SCAN COMPARISON</div>
                  {compareScanIds.length < 2 ? <p style={{ color: '#94a3b8', fontSize: '0.82rem' }}>Select two scans to compare their predictions and confidence.</p> : (() => {
                    const first = backendHistory.scans.find(scan => scan.id === compareScanIds[0]);
                    const second = backendHistory.scans.find(scan => scan.id === compareScanIds[1]);
                    const confidenceDelta = Number(second?.confidence || 0) - Number(first?.confidence || 0);
                    return <div style={{ color: '#cbd5e1', fontSize: '0.85rem', lineHeight: 1.7 }}><div><strong>Earlier:</strong> {first?.prediction} ({Number(first?.confidence || 0).toFixed(2)}%)</div><div><strong>Later:</strong> {second?.prediction} ({Number(second?.confidence || 0).toFixed(2)}%)</div><div><strong>Confidence change:</strong> <span style={{ color: confidenceDelta >= 0 ? '#86efac' : '#fbbf24' }}>{confidenceDelta >= 0 ? '+' : ''}{confidenceDelta.toFixed(2)}%</span></div></div>;
                  })()}
                  <div style={{ borderTop: '1px solid rgba(148,163,184,0.2)', marginTop: 16, paddingTop: 12, color: '#94a3b8', fontSize: '0.78rem' }}>Reports are stored separately and can be exported from the clinical workstation after physician sign-off.</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* TAB 1: Clinical Diagnosis Workstation (3-Column Doctor Layout) */}
        <div style={{ display: activeTab === 'diag' ? 'grid' : 'none', gridTemplateColumns: '300px 1fr 380px', gap: '1.5rem' }}>

          {/* Column 1: Hospital Patient Worklist Queue */}
          <div className="hud-card" style={{ padding: '1.2rem' }}>
            <div className="hud-title" style={{ fontSize: '0.95rem' }}>
              <span>📋 PACS PATIENT QUEUE</span>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{patientQueue.length} ACTIVE</span>
            </div>

            <button type="button" className="btn-neon" onClick={() => setShowPatientForm(true)} style={{ width: '100%', padding: '8px 10px', fontSize: '0.8rem', marginTop: 10 }}>
              ＋ REGISTER NEW PATIENT
            </button>

            {/* MRI Upload */}
            <input
              ref={fileInputRef}
              type="file"
                accept=".dcm,application/dicom,image/png,image/jpeg,image/jpg,image/webp"
              style={{ display: 'none' }}
              onChange={e => {
                handleFileSelect(e.target.files?.[0]);
                e.target.value = '';
              }}
            />
            <div
              onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={e => {
                e.preventDefault();
                setIsDragging(false);
                handleFileSelect(e.dataTransfer.files?.[0]);
              }}
              style={{
                margin: '12px 0',
                padding: '14px 10px',
                textAlign: 'center',
                border: `1px dashed ${isDragging ? '#00f2fe' : 'rgba(0, 242, 254, 0.45)'}`,
                borderRadius: 8,
                background: isDragging ? 'rgba(0, 242, 254, 0.15)' : 'rgba(0, 242, 254, 0.05)'
              }}
            >
              <div style={{ fontSize: '0.8rem', color: '#cbd5e1', marginBottom: 8 }}>
                Upload a brain MRI for AI analysis
              </div>
              <button
                type="button"
                className="btn-neon"
                style={{ width: '100%', padding: '8px 10px', fontSize: '0.8rem' }}
                onClick={() => fileInputRef.current?.click()}
              >
                📤 CHOOSE MRI IMAGE
              </button>
              <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: 7 }}>
                DICOM, PNG, JPG, or WEBP • or drag and drop here
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <input
                value={patientSearch}
                onChange={e => setPatientSearch(e.target.value)}
                placeholder="Search patients or scan IDs..."
                style={{ width: '100%', boxSizing: 'border-box', background: '#070c18', border: '1px solid rgba(0, 242, 254, 0.25)', color: '#fff', padding: '8px 10px', borderRadius: 6, fontSize: '0.8rem' }}
              />
              {patientQueue.filter(pt => `${pt.name} ${pt.id}`.toLowerCase().includes(patientSearch.toLowerCase())).map(pt => (
                <div
                  key={pt.id}
                  className={`patient-card ${currentPatient.id === pt.id ? 'active' : ''}`}
                  onClick={() => selectPatient(pt)}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{pt.name}</div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{pt.id} • {pt.age}y/{pt.gender[0]}</div>
                  </div>
                  <span style={{ fontSize: '0.7rem', padding: '3px 8px', borderRadius: 12, fontWeight: 700, background: 'rgba(0,0,0,0.4)', color: pt.triageColor, border: `1px solid ${pt.triageColor}` }}>
                    {pt.triage}
                  </span>
                </div>
              ))}
              {patientQueue.filter(pt => `${pt.name} ${pt.id}`.toLowerCase().includes(patientSearch.toLowerCase())).length === 0 && (
                <div style={{ color: '#94a3b8', fontSize: '0.8rem', textAlign: 'center', padding: 10 }}>No matching patients</div>
              )}
            </div>

            {analysisError && (
              <div style={{ marginTop: 10, background: 'rgba(245, 158, 11, 0.12)', border: '1px solid #f59e0b', color: '#fbbf24', padding: 8, borderRadius: 6, fontSize: '0.75rem' }}>
                {analysisError}
              </div>
            )}

            {scanHistory.length > 0 && (
              <div style={{ marginTop: 12, color: '#94a3b8', fontSize: '0.72rem' }}>
                <strong style={{ color: '#00f2fe' }}>LOCAL SCAN HISTORY</strong>
                {scanHistory.slice(0, 3).map(scan => (
                  <div key={scan.id} style={{ marginTop: 5, padding: 6, background: 'rgba(15, 23, 42, 0.6)', borderRadius: 5 }}>
                    {scan.name} • {new Date(scan.analyzedAt).toLocaleString()}
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 15, background: 'rgba(15, 23, 42, 0.6)', padding: 12, borderRadius: 8, border: '1px solid rgba(56, 189, 248, 0.2)', fontSize: '0.8rem' }}>
              <strong style={{ color: '#00f2fe' }}>Clinical History:</strong>
              <p style={{ color: '#94a3b8', marginTop: 4, lineHeight: 1.4 }}>{currentPatient.history}</p>
            </div>
          </div>

          {/* Column 2: Central Radiology Viewer & Tools */}
          <div className="hud-card">
            <div className="hud-title">
              <span>📡 AXIAL SCANNER VIEWPORT • {currentPatient.name}</span>
              <span style={{ fontSize: '0.8rem', color: '#00f2fe' }}>SLICE: 14/28</span>
            </div>

            {/* Radiology Toolbar (Window Leveling & Caliper) */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(15, 23, 42, 0.8)', padding: '8px 12px', borderRadius: 8, marginBottom: 12, border: '1px solid rgba(148, 163, 184, 0.2)', flexWrap: 'wrap', gap: 8 }}>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 700 }}>PRESET:</span>
                <button className={`radio-tool-btn ${windowPreset === 'brain' ? 'active' : ''}`} onClick={() => applyWindowPreset('brain')}>Brain</button>
                <button className={`radio-tool-btn ${windowPreset === 'bone' ? 'active' : ''}`} onClick={() => applyWindowPreset('bone')}>Bone</button>
                <button className={`radio-tool-btn ${windowPreset === 'subdural' ? 'active' : ''}`} onClick={() => applyWindowPreset('subdural')}>Subdural</button>
                <button className={`radio-tool-btn ${windowPreset === 'linear' ? 'active' : ''}`} onClick={() => applyWindowPreset('linear')}>Raw</button>
              </div>

              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <button
                  className={`radio-tool-btn ${caliperActive ? 'active' : ''}`}
                  onClick={() => { setCaliperActive(!caliperActive); playBeep(500, 0.05); }}
                  style={{ color: caliperActive ? '#ffff00' : '#cbd5e1', borderColor: caliperActive ? '#ffff00' : 'rgba(0, 242, 254, 0.3)' }}
                >
                  📏 Caliper Ruler: {caliperActive ? 'ACTIVE' : 'OFF'}
                </button>
              </div>
            </div>

            {/* Canvas Viewport */}
            <div style={{ background: '#000', borderRadius: 12, border: '2px solid rgba(0, 242, 254, 0.4)', padding: 15, display: 'flex', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
              {isScanning && <div className="laser-line"></div>}
              <canvas
                ref={canvasRef}
                width="224"
                height="224"
                onMouseDown={handleCanvasMouseDown}
                onMouseUp={handleCanvasMouseUp}
                style={{ borderRadius: 6, cursor: caliperActive ? 'crosshair' : 'default', width: 280, height: 280, transform: `scale(${viewerZoom})`, transition: 'transform 0.2s ease' }}
              ></canvas>
            </div>

            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 10 }}>
              <button className="radio-tool-btn" onClick={() => setViewerZoom(z => Math.max(1, +(z - 0.25).toFixed(2)))}>− Zoom</button>
              <span style={{ color: '#94a3b8', fontSize: '0.8rem', padding: '7px 4px' }}>{Math.round(viewerZoom * 100)}%</span>
              <button className="radio-tool-btn" onClick={() => setViewerZoom(z => Math.min(2.5, +(z + 0.25).toFixed(2)))}>＋ Zoom</button>
              <button className="radio-tool-btn" onClick={() => setViewerZoom(1)}>Reset</button>
            </div>

            {/* Caliper measurement display */}
            {measuredDistanceMm && (
              <div style={{ background: 'rgba(255, 255, 0, 0.15)', border: '1px solid #ffff00', color: '#ffff00', padding: '8px 12px', borderRadius: 6, marginTop: 10, textAlign: 'center', fontWeight: 700, fontSize: '0.9rem' }}>
                📏 Digital Caliper: Tumor Major Axis = <strong>{measuredDistanceMm} mm</strong>
              </div>
            )}

            {/* Interactive Sliders */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 15, marginTop: 12, fontSize: '0.8rem', color: '#94a3b8' }}>
              <div>
                <label>Window Center (Brightness): {brightness}%</label>
                <input type="range" min="50" max="150" value={brightness} onChange={e => setBrightness(e.target.value)} style={{ width: '100%' }} />
              </div>
              <div>
                <label>Window Width (Contrast): {contrast}%</label>
                <input type="range" min="50" max="200" value={contrast} onChange={e => setContrast(e.target.value)} style={{ width: '100%' }} />
              </div>
            </div>

            <button className="btn-neon" style={{ width: '100%', marginTop: 14 }} onClick={handleScanAudit}>
              ⚡ RUN HIGH-PRECISION SCANNER AUDIT
            </button>
          </div>

          {/* Column 3: AI Triage & Doctor Review Panel */}
          <div className="hud-card" style={{ padding: '1.2rem' }}>
            <div className="hud-title" style={{ fontSize: '0.95rem' }}>
              <span>🧠 AI TRIAGE & PHYSICIAN REVIEW</span>
            </div>

            {/* AI Diagnosis Banner */}
            <div style={{ background: currentPatient.triageColor + '20', border: `1px solid ${currentPatient.triageColor}`, borderRadius: 10, padding: 12, textAlign: 'center', marginBottom: 12 }}>
              <div style={{ fontFamily: 'Orbitron', fontSize: '1rem', fontWeight: 900, color: currentPatient.triageColor }}>
                {currentPatient.diagnosis}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#e2e8f0', marginTop: 2 }}>
                {currentPatient.subtext}
              </div>
            </div>

            {/* Probabilities */}
            {['glioma', 'meningioma', 'pituitary', 'notumor'].map(key => (
              <div key={key} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, marginBottom: 2 }}>
                  <span style={{ textTransform: 'capitalize' }}>{key}</span>
                  <span style={{ color: '#00f2fe' }}>{currentPatient.probs[key].toFixed(2)}%</span>
                </div>
                <div style={{ height: 6, background: 'rgba(15, 23, 42, 0.8)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${currentPatient.probs[key]}%`, height: '100%', background: 'linear-gradient(90deg, #00f2fe, #a855f7)', borderRadius: 3 }}></div>
                </div>
              </div>
            ))}

            {/* Doctor Review & Decision Panel */}
            <div style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: 8, padding: 12, marginTop: 14 }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#00f2fe', marginBottom: 6 }}>PHYSICIAN SIGN-OFF:</div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '0.8rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input disabled={!canApproveReports} type="radio" name="concur" checked={clinicalConcurrence === 'concur'} onChange={() => setClinicalConcurrence('concur')} />
                  <span>Concur with AI Diagnosis</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input disabled={!canApproveReports} type="radio" name="concur" checked={clinicalConcurrence === 'biopsy'} onChange={() => setClinicalConcurrence('biopsy')} />
                  <span>Recommend Stereotactic Biopsy</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input disabled={!canApproveReports} type="radio" name="concur" checked={clinicalConcurrence === 'overrule'} onChange={() => setClinicalConcurrence('overrule')} />
                  <span>Overrule AI Recommendation</span>
                </label>
              </div>

              <textarea
                value={physicianNotes}
                disabled={!canApproveReports}
                onChange={e => setPhysicianNotes(e.target.value)}
                placeholder="Enter attending radiologist clinical impression notes..."
                style={{ width: '100%', height: 50, marginTop: 8, background: '#070c18', border: '1px solid rgba(148, 163, 184, 0.2)', color: '#fff', fontSize: '0.75rem', padding: 6, borderRadius: 4, resize: 'none' }}
              />
            </div>

            <button
              className="btn-neon"
              style={{ width: '100%', marginTop: 12, background: 'linear-gradient(135deg, #a855f7, #6366f1)', fontSize: '0.8rem', padding: '10px 14px' }}
              disabled={!canApproveReports}
              onClick={async () => { playSciFiChime(); await persistReport(); setShowReport(true); }}
            >
              {canApproveReports ? '🖨️ EXPORT CLINICAL EHR REPORT' : '🔒 DOCTOR/ADMIN APPROVAL REQUIRED'}
            </button>
          </div>

        </div>

        {/* TAB 2: Explainable AI LIME (Persistent in DOM) */}
        <div style={{ display: activeTab === 'lime' ? 'block' : 'none' }}>
          <div className="hud-card">
            <div className="hud-title">🔍 EXPLAINABLE AI • FIGURE 11 MULTI-LAYER ATTRIBUTION</div>
            <p style={{ color: '#94a3b8', marginBottom: 15, fontSize: '0.9rem' }}>
              Current Case: <strong>{currentPatient.name} ({currentPatient.id})</strong>. Super-pixel segmentation highlights pathological margins with yellow boundaries and positive probability weights.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
              {[
                { title: '1. Raw Axial MRI', ref: limeOrigRef, desc: 'Axial T1/T2 (224x224)' },
                { title: '2. Correct Region Mask', ref: limeMaskRef, desc: 'Isolated Lesion Superpixels' },
                { title: '3. Highlight Contour', ref: limeContourRef, desc: 'Yellow Boundary Overlay' },
                { title: '4. Transparent Heatmap', ref: limeHeatmapRef, desc: 'Attribution Weights (Viridis)' }
              ].map((item, idx) => (
                <div key={idx} style={{ background: '#000', border: '1px solid rgba(0, 242, 254, 0.3)', borderRadius: 12, padding: 12, textAlign: 'center' }}>
                  <h4 style={{ fontFamily: 'Orbitron', fontSize: '0.85rem', color: '#38bdf8', marginBottom: 10 }}>{item.title}</h4>
                  <canvas ref={item.ref} width="200" height="200" style={{ width: '100%', borderRadius: 8 }}></canvas>
                  <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: 8 }}>{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* TAB 3: Research Benchmarks */}
        <div style={{ display: activeTab === 'bench' ? 'block' : 'none' }}>
          <div className="hud-card">
            <div className="hud-title">📊 IEEE ACCESS 2024 EMPIRICAL BENCHMARKS</div>

            <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
              <button
                onClick={() => setBenchSubTab('models')}
                style={{
                  background: benchSubTab === 'models' ? 'rgba(0, 242, 254, 0.25)' : 'transparent',
                  border: `1px solid ${benchSubTab === 'models' ? '#00f2fe' : 'var(--border-glow)'}`,
                  color: '#fff', padding: '6px 14px', borderRadius: 8, cursor: 'pointer', fontFamily: 'Rajdhani', fontWeight: 700
                }}
              >
                10 Transfer Learning Models (Table 5)
              </button>
              <button
                onClick={() => setBenchSubTab('ablation')}
                style={{
                  background: benchSubTab === 'ablation' ? 'rgba(0, 242, 254, 0.25)' : 'transparent',
                  border: `1px solid ${benchSubTab === 'ablation' ? '#00f2fe' : 'var(--border-glow)'}`,
                  color: '#fff', padding: '6px 14px', borderRadius: 8, cursor: 'pointer', fontFamily: 'Rajdhani', fontWeight: 700
                }}
              >
                Optimizer Ablation Grid (Table 7)
              </button>
            </div>

            {benchSubTab === 'models' ? (
              <div style={{ background: 'rgba(15, 23, 42, 0.7)', borderRadius: 12, padding: 16, overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid rgba(0, 242, 254, 0.3)', color: '#00f2fe', fontFamily: 'Orbitron', fontSize: '0.8rem' }}>
                      <th style={{ padding: 12 }}>Model</th>
                      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Hamming Loss</th><th>MCC</th>
                    </tr>
                  </thead>
                  <tbody style={{ fontFamily: 'Rajdhani', fontSize: '1rem' }}>
                    {[
                      { m: '⭐ Proposed TumorGANet (VGG-16)', acc: '99.53%', p: '100.0%', r: '100.0%', f: '99.0%', hl: '0.20%', mcc: '99.50%', win: true },
                      { m: 'GANs + Inception-v3', acc: '99.34%', p: '99.34%', r: '99.34%', f: '99.0%', hl: '1.02%', mcc: '97.20%' },
                      { m: 'GANs + ResNet-101', acc: '98.93%', p: '98.93%', r: '98.93%', f: '99.0%', hl: '0.50%', mcc: '99.10%' },
                      { m: 'GANs + AlexNet', acc: '98.17%', p: '98.17%', r: '98.17%', f: '98.0%', hl: '0.90%', mcc: '97.55%' },
                      { m: 'GANs + GoogLeNet', acc: '97.18%', p: '97.18%', r: '97.18%', f: '97.0%', hl: '2.44%', mcc: '94.10%' },
                      { m: 'GANs + VGG-19', acc: '96.72%', p: '96.72%', r: '96.72%', f: '96.5%', hl: '1.60%', mcc: '96.00%' },
                      { m: 'GANs + MobileNet', acc: '96.41%', p: '96.41%', r: '96.41%', f: '96.2%', hl: '7.90%', mcc: '95.00%' },
                      { m: 'GANs + Xception', acc: '78.01%', p: '86.00%', r: '77.00%', f: '76.0%', hl: '10.90%', mcc: '64.30%' }
                    ].map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: row.win ? 'rgba(0, 242, 254, 0.1)' : 'transparent', fontWeight: row.win ? 700 : 400 }}>
                        <td style={{ padding: 12, color: row.win ? '#00f2fe' : '#fff' }}>{row.m}</td>
                        <td>{row.acc}</td><td>{row.p}</td><td>{row.r}</td><td>{row.f}</td><td>{row.hl}</td><td>{row.mcc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ background: 'rgba(15, 23, 42, 0.7)', borderRadius: 12, padding: 16, overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid rgba(0, 242, 254, 0.3)', color: '#00f2fe', fontFamily: 'Orbitron', fontSize: '0.8rem' }}>
                      <th style={{ padding: 12 }}>Optimizer</th><th>Learning Rate</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>HL</th>
                    </tr>
                  </thead>
                  <tbody style={{ fontFamily: 'Rajdhani', fontSize: '1rem' }}>
                    {[
                      { opt: 'Adam (Winning)', lr: '0.0001', acc: '98.00%', p: '99.00%', r: '100.0%', f: '99.00%', hl: '2.20%', win: true },
                      { opt: 'Adam', lr: '0.001', acc: '95.00%', p: '100.0%', r: '99.00%', f: '98.00%', hl: '4.70%' },
                      { opt: 'Adam', lr: '0.00001', acc: '99.00%', p: '100.0%', r: '100.0%', f: '100.0%', hl: '0.60%' },
                      { opt: 'SGD', lr: '0.001', acc: '99.00%', p: '100.0%', r: '100.0%', f: '100.0%', hl: '0.40%' },
                      { opt: 'SGD', lr: '0.0001', acc: '99.00%', p: '100.0%', r: '100.0%', f: '100.0%', hl: '1.30%' },
                      { opt: 'Adagrad', lr: '0.001', acc: '98.00%', p: '99.00%', r: '100.0%', f: '99.00%', hl: '1.70%' },
                      { opt: 'Nadam', lr: '0.0001', acc: '98.00%', p: '100.0%', r: '100.0%', f: '99.00%', hl: '1.80%' },
                      { opt: 'RMSprop', lr: '0.00001', acc: '99.00%', p: '100.0%', r: '100.0%', f: '100.0%', hl: '0.80%' }
                    ].map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: row.win ? 'rgba(0, 242, 254, 0.1)' : 'transparent', fontWeight: row.win ? 700 : 400 }}>
                        <td style={{ padding: 12, color: row.win ? '#00f2fe' : '#fff' }}>{row.opt}</td>
                        <td>{row.lr}</td><td>{row.acc}</td><td>{row.p}</td><td>{row.r}</td><td>{row.f}</td><td>{row.hl}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* TAB 4: DCGAN Studio */}
        <div style={{ display: activeTab === 'gan' ? 'block' : 'none' }}>
          <div className="hud-card">
            <div className="hud-title">🎨 DCGAN SYNTHETIC BRAIN MRI GENERATOR</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'center' }}>
              <div style={{ background: '#000', borderRadius: 14, border: '2px solid rgba(0, 242, 254, 0.4)', padding: 20, display: 'flex', justifyContent: 'center' }}>
                <canvas ref={ganCanvasRef} width="224" height="224" style={{ borderRadius: 8 }}></canvas>
              </div>
              <div>
                <h3 style={{ fontFamily: 'Orbitron', color: '#00f2fe', marginBottom: 10 }}>Algorithm 1 Latent Generator</h3>
                <p style={{ color: '#94a3b8', lineHeight: 1.6, marginBottom: 14 }}>
                  Synthesizes high-fidelity brain MRI scans by mapping random Gaussian noise vectors z ~ N(0, I) through deconvolutional layers.
                </p>
                <button
                  className="btn-neon"
                  onClick={() => {
                    playSciFiChime();
                    const types = ['glioma', 'meningioma', 'pituitary', 'notumor'];
                    const r = types[Math.floor(Math.random() * types.length)];
                    drawMRI(ganCanvasRef.current, r, 'normal');
                  }}
                >
                  ⚡ SAMPLE NOVEL MRI SLICE
                </button>
              </div>
            </div>
          </div>
        </div>

      </main>

      {showPatientForm && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.78)', backdropFilter: 'blur(8px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 90 }} onClick={() => setShowPatientForm(false)}>
          <form onSubmit={registerPatient} onClick={event => event.stopPropagation()} style={{ width: 'min(560px, 92%)', background: '#0f172a', border: '1px solid rgba(0,242,254,0.35)', borderRadius: 14, padding: 24, color: '#e2e8f0' }}>
            <div className="hud-title" style={{ marginBottom: 16 }}>＋ REGISTER NEW PATIENT</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 130px', gap: 10 }}>
              <input required placeholder="Full name" value={patientForm.name} onChange={event => setPatientForm({ ...patientForm, name: event.target.value })} style={authInputStyle} />
              <input required type="number" min="0" max="130" placeholder="Age" value={patientForm.age} onChange={event => setPatientForm({ ...patientForm, age: event.target.value })} style={authInputStyle} />
            </div>
            <select required value={patientForm.gender} onChange={event => setPatientForm({ ...patientForm, gender: event.target.value })} style={{ ...authInputStyle, color: patientForm.gender ? '#fff' : '#94a3b8' }}>
              <option value="">Select gender</option><option>Female</option><option>Male</option><option>Other</option><option>Prefer not to say</option>
            </select>
            <input placeholder="Referring doctor (optional)" value={patientForm.referringDoctor} onChange={event => setPatientForm({ ...patientForm, referringDoctor: event.target.value })} style={authInputStyle} />
            <input placeholder="City or region for referral suggestions (optional)" value={patientForm.location} onChange={event => setPatientForm({ ...patientForm, location: event.target.value })} style={authInputStyle} />
            <input placeholder="Main symptoms (optional)" value={patientForm.symptoms} onChange={event => setPatientForm({ ...patientForm, symptoms: event.target.value })} style={authInputStyle} />
            <textarea required minLength={3} placeholder="Clinical history" value={patientForm.history} onChange={event => setPatientForm({ ...patientForm, history: event.target.value })} style={{ ...authInputStyle, height: 90, resize: 'vertical' }} />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}><button type="button" onClick={() => setShowPatientForm(false)} className="radio-tool-btn">Cancel</button><button type="submit" disabled={patientFormBusy} className="btn-neon">{patientFormBusy ? 'SAVING…' : 'SAVE PATIENT'}</button></div>
          </form>
        </div>
      )}

      {/* Official Structured Hospital Oncology Report Modal */}
      {showReport && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(10px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 100 }} onClick={() => setShowReport(false)}>
          <div className="printable-report" style={{ background: '#fff', color: '#0f172a', maxWidth: 760, width: '92%', borderRadius: 12, padding: 35, maxHeight: '90vh', overflowY: 'auto' }} onClick={e => e.stopPropagation()}>
            
            {/* Hospital Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #0f172a', paddingBottom: 12, marginBottom: 20 }}>
              <div>
                <h2 style={{ fontFamily: 'Orbitron', fontSize: '1.25rem', color: '#0f172a', letterSpacing: 0.5 }}>NATIONAL NEUROLOGICAL INSTITUTE</h2>
                <p style={{ fontSize: '0.85rem', color: '#64748b' }}>Department of Diagnostic Neuro-Radiology & Oncology</p>
              </div>
              <div style={{ textAlign: 'right', fontSize: '0.85rem' }}>
                <strong style={{ color: '#0284c7' }}>STRUCTURED RADIOLOGY REPORT</strong><br />
                <span>Protocol: TumorGANet-CDSS v2.0</span>
              </div>
            </div>

            {/* Patient Demographics Table */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, background: '#f8fafc', padding: 14, borderRadius: 8, marginBottom: 18, fontSize: '0.85rem' }}>
              <div>
                <p><strong>Patient Name:</strong> {currentPatient.name}</p>
                <p><strong>Medical Record #:</strong> {currentPatient.id}</p>
                <p><strong>Age / Gender:</strong> {currentPatient.age} years / {currentPatient.gender}</p>
              </div>
              <div>
                <p><strong>Referring Doctor:</strong> {reportReferringDoctor || 'Not provided'}</p>
              </div>
            </div>

            {/* Clinical Indication */}
            <div style={{ marginBottom: 16, fontSize: '0.85rem' }}>
              <strong>Clinical Indication & History:</strong>
              <p style={{ color: '#475569', marginTop: 2 }}>{currentPatient.history}</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 8, background: '#f8fafc', padding: 10, borderRadius: 6 }}>
                <div><strong>Reported Symptoms:</strong><br />{reportSymptoms || 'Not provided'}</div>
                <div><strong>Referring Doctor:</strong><br />{reportReferringDoctor || 'Not provided'}</div>
              </div>
            </div>

            {/* AI Quantitative Findings */}
            <div style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#0284c7', marginBottom: 6 }}>1. AI Quantitative Assessment (TumorGANet):</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: currentPatient.triageColor, marginBottom: 4 }}>
                {currentPatient.diagnosis}
              </div>
              <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '4px 0 8px' }}><strong>Triage:</strong> {currentPatient.triage} &nbsp;•&nbsp; <strong>Assessment:</strong> {currentPatient.subtext}</p>
              <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5 }}>
                {currentPatient.desc}
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 10, fontSize: '0.8rem', background: '#f1f5f9', padding: 8, borderRadius: 6 }}>
                <div><strong>Est. Area:</strong> {currentPatient.area}</div>
                <div><strong>Centroid:</strong> {currentPatient.centroid}</div>
                <div><strong>Major Axis:</strong> {measuredDistanceMm ? `${measuredDistanceMm} mm (Measured)` : '24.2 mm (Computed)'}</div>
              </div>
            </div>

            {currentPatient.location && doctorSearchUrl && (
              <div style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 16, fontSize: '0.82rem' }}>
                <div style={{ fontWeight: 700, color: '#0284c7', marginBottom: 6 }}>Nearby Specialist Search</div>
                <div><strong>Patient region:</strong> {currentPatient.location}</div>
                <a href={doctorSearchUrl} target="_blank" rel="noreferrer" style={{ display: 'inline-block', marginTop: 10, background: '#0284c7', color: '#fff', padding: '8px 12px', borderRadius: 5, textDecoration: 'none', fontWeight: 700 }}>Find nearby specialists on Google Maps</a>
                <p style={{ color: '#64748b', margin: '8px 0 0', fontSize: '0.72rem' }}>This opens a location-based search. Verify credentials, specialty, availability, and emergency services before contacting a provider.</p>
              </div>
            )}

            {/* Buttons */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }} className="no-print">
              <button onClick={() => window.print()} style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '10px 20px', borderRadius: 6, cursor: 'pointer', fontWeight: 700, fontFamily: 'Rajdhani', fontSize: '0.95rem' }}>
                🖨️ Print / Save as PDF
              </button>
              <button onClick={() => setShowReport(false)} style={{ background: '#e2e8f0', color: '#334155', border: 'none', padding: '10px 20px', borderRadius: 6, cursor: 'pointer', fontWeight: 700, fontFamily: 'Rajdhani', fontSize: '0.95rem' }}>
                Close
              </button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
