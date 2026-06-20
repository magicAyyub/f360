// ── Constants 
const PW = 680, PH = 440, PAD = 36;
const FW = PW - PAD*2, FH = PH - PAD*2;

// Colors as [r, g, b]
const C_TEAM_A = [55,  138, 221];
const C_TEAM_B = [239, 159, 39 ];
const C_REF    = [99,  153, 34 ];
const C_BALL   = [226, 75,  74 ];
const C_GREEN  = [45,  106, 45 ];

// ── State 
let TRAIL_LEN = 15;
let simFrame  = 0;
let playing   = true;
const HISTORY = {};
const tip = document.getElementById('tip');
let ball, players;

// Normalised → canvas coordinate helpers
function sx(v) { return PAD + v * FW; }
function sy(v) { return PAD + v * FH; }

// ── Simulation 
function initPlayers() {
  const ps = [];
  const fA = [[0.12,0.5],[0.25,0.2],[0.25,0.5],[0.25,0.8],
              [0.4,0.15],[0.4,0.38],[0.4,0.62],[0.4,0.85],
              [0.55,0.25],[0.55,0.5],[0.55,0.75]];
  const fB = [[0.88,0.5],[0.75,0.2],[0.75,0.5],[0.75,0.8],
              [0.6,0.15],[0.6,0.38],[0.6,0.62],[0.6,0.85],
              [0.45,0.25],[0.45,0.5],[0.45,0.75]];
  fA.forEach((p,i) => ps.push({id:`A${i+1}`,team:0,x:p[0],y:p[1],vx:(Math.random()-.5)*0.002,vy:(Math.random()-.5)*0.002,num:i+1}));
  fB.forEach((p,i) => ps.push({id:`B${i+1}`,team:1,x:p[0],y:p[1],vx:(Math.random()-.5)*0.002,vy:(Math.random()-.5)*0.002,num:i+1}));
  ps.push({id:'R1',team:2,x:0.5, y:0.05,vx: 0.001,vy:0.001, num:'R'});
  ps.push({id:'R2',team:2,x:0.3, y:0.95,vx:-0.001,vy:0.001, num:'R'});
  return ps;
}

function initState() {
  ball    = {x:0.5, y:0.5, vx:0.006, vy:0.004};
  players = initPlayers();
  Object.keys(HISTORY).forEach(k => delete HISTORY[k]);
  simFrame = 0;
}

function simulateStep() {
  ball.x += ball.vx; ball.y += ball.vy;
  if (ball.x < 0.01 || ball.x > 0.99) { ball.vx *= -1; ball.x = constrain(ball.x, 0.01, 0.99); }
  if (ball.y < 0.01 || ball.y > 0.99) { ball.vy *= -1; ball.y = constrain(ball.y, 0.01, 0.99); }
  if (Math.random() < 0.015) { ball.vx += (Math.random()-.5)*0.008; ball.vy += (Math.random()-.5)*0.008; }
  const bspd = Math.sqrt(ball.vx**2 + ball.vy**2);
  if (bspd > 0.018) { ball.vx *= 0.85; ball.vy *= 0.85; }

  players.forEach(p => {
    const tgt = p.team===2
      ? {x:0.5, y:0.5}
      : (Math.random()<0.3 ? ball : {x:p.x+Math.random()*.1-.05, y:p.y+Math.random()*.1-.05});
    const dx = tgt.x-p.x, dy = tgt.y-p.y, d = Math.sqrt(dx*dx+dy*dy)||1;
    const sp = p.team===2 ? 0.0008 : 0.0015;
    p.vx = p.vx*0.8 + (dx/d)*sp*Math.random()*2;
    p.vy = p.vy*0.8 + (dy/d)*sp*Math.random()*2;
    p.x  = constrain(p.x + p.vx, 0.02, 0.98);
    p.y  = constrain(p.y + p.vy, 0.02, 0.98);
    if (!HISTORY[p.id]) HISTORY[p.id] = [];
    HISTORY[p.id].push({x:p.x, y:p.y});
    if (HISTORY[p.id].length > 60) HISTORY[p.id].shift();
  });
  if (!HISTORY['ball']) HISTORY['ball'] = [];
  HISTORY['ball'].push({x:ball.x, y:ball.y});
  if (HISTORY['ball'].length > 60) HISTORY['ball'].shift();
}

function closestToBall() {
  let minD = Infinity, closest = null;
  players.filter(p => p.team < 2).forEach(p => {
    const d = (p.x-ball.x)**2 + (p.y-ball.y)**2;
    if (d < minD) { minD = d; closest = p; }
  });
  return closest;
}

// ── p5.js sketch 
function setup() {
  const cnv = createCanvas(PW, PH);
  cnv.parent('pitch-wrap');
  textFont('sans-serif');
  initState();
}

function draw() {
  if (playing) { simulateStep(); simFrame++; }

  background(C_GREEN[0], C_GREEN[1], C_GREEN[2]);
  drawPitch();
  drawTrails();
  drawPlayers();
  drawBall();
  updateHUD();
}

// ── Pitch 
function drawPitch() {
  const LW = 1.5;

  // Zone stripes (subtle)
  noStroke();
  for (let i = 1; i < 5; i++) {
    fill(255, 255, 255, 15);
    rect(PAD + FW/5*i - 0.5, PAD, 1, FH);
  }

  stroke(255, 255, 255, 217);
  strokeWeight(LW);
  noFill();

  // Outer boundary & halfway line
  rect(PAD, PAD, FW, FH);
  line(PAD+FW/2, PAD, PAD+FW/2, PAD+FH);

  // Center circle
  circle(PAD+FW/2, PAD+FH/2, FH*0.3); // p5 circle takes diameter

  // Penalty areas & goal areas
  const bw = FW*0.13, bh = FH*0.38, sw = FW*0.04, sh = FH*0.16;
  rect(PAD,       PAD+FH/2-bh/2, bw, bh);
  rect(PAD,       PAD+FH/2-sh/2, sw, sh);
  rect(PAD+FW-bw, PAD+FH/2-bh/2, bw, bh);
  rect(PAD+FW-sw, PAD+FH/2-sh/2, sw, sh);

  // Penalty arcs (D-shapes)
  // p5 arc angles: 0 = right (east), clockwise. OPEN = arc stroke only.
  const ar  = FH * 0.09;
  const psd = FW * 0.1;
  const arcT = Math.acos(Math.min(1, (FW*0.13 - psd) / ar));
  arc(PAD + psd,     PAD+FH/2, ar*2, ar*2, -arcT,   arcT,    OPEN);
  arc(PAD+FW - psd,  PAD+FH/2, ar*2, ar*2, PI-arcT, PI+arcT, OPEN);

  // Goals
  const gw = FW*0.015, gh = FH*0.14;
  rect(PAD-gw,  PAD+FH/2-gh/2, gw, gh);
  rect(PAD+FW,  PAD+FH/2-gh/2, gw, gh);

  // Corner arcs (quarter-circles curving into the field)
  const cr = FH * 0.04;
  arc(PAD,    PAD,    cr*2, cr*2, 0,       HALF_PI,          OPEN); // top-left
  arc(PAD+FW, PAD,    cr*2, cr*2, HALF_PI, PI,               OPEN); // top-right
  arc(PAD,    PAD+FH, cr*2, cr*2, -HALF_PI, 0,               OPEN); // bottom-left
  arc(PAD+FW, PAD+FH, cr*2, cr*2, PI,      PI + HALF_PI,     OPEN); // bottom-right

  // Spots (center, penalty)
  fill(255, 255, 255, 217);
  noStroke();
  circle(PAD+FW/2,    PAD+FH/2, 6); // center spot (diameter 6 = radius 3)
  circle(PAD + psd,   PAD+FH/2, 5); // left penalty spot
  circle(PAD+FW - psd, PAD+FH/2, 5); // right penalty spot
}

// ── Trails 
function drawTrails() {
  noFill();

  // Ball trail
  const bh = (HISTORY['ball'] || []).slice(-TRAIL_LEN);
  if (bh.length >= 2) {
    strokeWeight(1.5);
    stroke(C_BALL[0], C_BALL[1], C_BALL[2], 153);
    beginShape();
    curveVertex(sx(bh[0].x), sy(bh[0].y));
    bh.forEach(pt => curveVertex(sx(pt.x), sy(pt.y)));
    curveVertex(sx(bh[bh.length-1].x), sy(bh[bh.length-1].y));
    endShape();
  }

  // Player trails
  players.forEach(p => {
    const h = (HISTORY[p.id] || []).slice(-TRAIL_LEN);
    if (h.length < 2) return;
    const c = p.team===0 ? C_TEAM_A : p.team===1 ? C_TEAM_B : C_REF;
    strokeWeight(1);
    stroke(c[0], c[1], c[2], 77);
    beginShape();
    curveVertex(sx(h[0].x), sy(h[0].y));
    h.forEach(pt => curveVertex(sx(pt.x), sy(pt.y)));
    curveVertex(sx(h[h.length-1].x), sy(h[h.length-1].y));
    endShape();
  });
}

// ── Players 
function drawPlayers() {
  const poss = closestToBall();
  textAlign(CENTER, CENTER);
  textSize(7);
  textStyle(BOLD);

  players.forEach(p => {
    const c  = p.team===0 ? C_TEAM_A : p.team===1 ? C_TEAM_B : C_REF;
    const r  = (poss && poss.id===p.id) ? 11 : 9;
    const cx = sx(p.x), cy = sy(p.y);

    fill(c[0], c[1], c[2]);
    stroke(255, 255, 255, 230);
    strokeWeight(1.5);
    circle(cx, cy, r * 2);

    fill(255); noStroke();
    text(p.num, cx, cy);
  });
}

// ── Ball 
function drawBall() {
  const cx = sx(ball.x), cy = sy(ball.y);
  // Outer glow
  noStroke(); fill(255, 255, 255, 50);
  circle(cx, cy, 14);
  // Ball
  fill(255); stroke(136, 136, 136); strokeWeight(1);
  circle(cx, cy, 10);
}

// ── HUD 
function updateHUD() {
  const poss = closestToBall();
  const spd  = (Math.sqrt(ball.vx**2 + ball.vy**2) * 3600 * 0.1).toFixed(1);
  document.getElementById('s-frame').textContent = simFrame;
  document.getElementById('s-speed').textContent = spd + ' km/h';
  document.getElementById('s-poss').textContent  = poss ? (poss.team===0 ? 'Équipe A' : 'Équipe B') : '—';
}

// ── Tooltip 
function mouseMoved() {
  if (!players) return;
  const cnv  = document.querySelector('#pitch-wrap canvas');
  const rect = cnv.getBoundingClientRect();
  // CSS-pixel position relative to canvas element
  const mx = winMouseX - rect.left;
  const my = winMouseY - rect.top;
  // Map to canvas coordinate space (handles CSS scaling)
  const cx = mx * (PW / rect.width);
  const cy = my * (PH / rect.height);

  let hovered = null;
  for (const p of players) {
    const dx = cx - sx(p.x), dy = cy - sy(p.y);
    if (dx*dx + dy*dy < 144) { hovered = p; break; }
  }

  if (hovered) {
    cursor(HAND);
    tip.style.opacity = '1';
    tip.style.left    = (mx + 12) + 'px';
    tip.style.top     = (my - 30) + 'px';
    const team = hovered.team===0 ? 'Équipe A' : hovered.team===1 ? 'Équipe B' : 'Arbitre';
    const sp   = (Math.sqrt(hovered.vx**2 + hovered.vy**2) * 3600 * 0.1).toFixed(1);
    tip.innerHTML = `<strong>${team} #${hovered.num}</strong> · ${sp} km/h`;
  } else {
    cursor(ARROW);
    tip.style.opacity = '0';
  }
}

// ── Controls 
function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {
    btn.innerHTML = '<i class="ti ti-player-pause" aria-hidden="true"></i> Pause';
    loop();
  } else {
    btn.innerHTML = '<i class="ti ti-player-play" aria-hidden="true"></i> Play';
    noLoop();
  }
}

function resetSim() {
  initState();
  playing = true;
  document.getElementById('btn-play').innerHTML = '<i class="ti ti-player-pause" aria-hidden="true"></i> Pause';
  loop();
}

// ── Video loading
function loadVideoFile(input) {
  if (!input.files || !input.files[0]) return;
  const url = URL.createObjectURL(input.files[0]);
  attachVideo(url);
}

function handleVideoDrop(event) {
  event.preventDefault();
  document.getElementById('video-area').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (!file || !file.type.startsWith('video/')) return;
  attachVideo(URL.createObjectURL(file));
}

function attachVideo(url) {
  const video = document.getElementById('match-video');
  const ph    = document.getElementById('video-ph');
  const badge = document.getElementById('sync-badge');
  video.src = url;
  video.style.display = 'block';
  ph.style.display    = 'none';
  badge.style.display = 'inline-block';
  // Remove click-to-open from the area once a video is loaded
  document.getElementById('video-area').onclick = null;
  document.getElementById('video-area').style.cursor = 'default';
}