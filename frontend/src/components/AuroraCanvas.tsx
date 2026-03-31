import { useEffect, useRef } from "react";

function drawAurora(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = 0.5 * H + (W * W) / (2.4 * H);
  const R = cy - 0.35 * H;

  const REF = 1200;
  const sc = Math.max(R / REF, 0.25);

  const sinH = Math.min((W * 0.55) / R, 0.9998);
  const halfA = Math.asin(sinH);
  const sA = -Math.PI / 2 - halfA;
  const eA = -Math.PI / 2 + halfA;

  const RAW: Array<[number, number, string, number]> = [
    [420, 340, "rgba(4,8,42,0.45)", 75],
    [330, 265, "rgba(7,18,78,0.54)", 60],
    [248, 195, "rgba(11,44,118,0.62)", 46],
    [175, 138, "rgba(17,78,156,0.69)", 35],
    [112, 90, "rgba(23,118,188,0.75)", 26],
    [60, 54, "rgba(32,155,208,0.81)", 18],
    [22, 28, "rgba(52,188,222,0.86)", 11],
    [7, 14, "rgba(110,212,232,0.91)", 6],
    [2, 7, "rgba(175,230,240,0.95)", 3],
    [0, 8, "rgba(248,245,214,0.9)", 2],
    [-7, 13, "rgba(255,238,170,0.97)", 4],
    [-22, 26, "rgba(255,208,68,0.93)", 10],
    [-46, 40, "rgba(255,160,22,0.90)", 17],
    [-80, 55, "rgba(252,104,8,0.87)", 24],
    [-124, 70, "rgba(222,56,7,0.83)", 32],
    [-176, 86, "rgba(170,26,6,0.78)", 41],
    [-238, 104, "rgba(110,12,4,0.70)", 50],
    [-308, 118, "rgba(58,6,3,0.58)", 58],
    [-388, 128, "rgba(25,3,2,0.42)", 66],
  ];

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  RAW.forEach(([rOff, lw, col, blur]) => {
    const r = R + rOff * sc;
    if (r < 1) return;

    ctx.save();
    ctx.filter = `blur(${Math.max(blur * Math.pow(sc, 0.55), 0.5).toFixed(1)}px)`;
    ctx.lineWidth = Math.max(lw * sc, 1.5);
    ctx.strokeStyle = col;
    ctx.lineCap = "butt";
    ctx.beginPath();
    ctx.arc(cx, cy, r, sA, eA);
    ctx.stroke();
    ctx.restore();
  });

  ctx.restore();

  const pSolid = Math.max(R - 460 * sc, 10);
  const pFade = Math.max(R - 255 * sc, 20);

  if (pFade > pSolid) {
    const pGrad = ctx.createRadialGradient(cx, cy, pSolid, cx, cy, pFade);
    pGrad.addColorStop(0, "rgba(7,6,15,1)");
    pGrad.addColorStop(0.6, "rgba(7,6,15,0.85)");
    pGrad.addColorStop(1, "rgba(7,6,15,0)");
    ctx.beginPath();
    ctx.arc(cx, cy, pFade, 0, Math.PI * 2);
    ctx.fillStyle = pGrad;
    ctx.fill();
  }

  const BG0 = "rgba(7,6,15,0)";
  const BG1 = "rgba(7,6,15,0.72)";

  const edgeWidth = W * 0.18;

  const lG = ctx.createLinearGradient(0, 0, edgeWidth, 0);
  lG.addColorStop(0, BG1);
  lG.addColorStop(0.22, BG1);
  lG.addColorStop(1, BG0);
  ctx.fillStyle = lG;
  ctx.fillRect(0, 0, edgeWidth, H);

  const rG = ctx.createLinearGradient(W - edgeWidth, 0, W, 0);
  rG.addColorStop(0, BG0);
  rG.addColorStop(0.78, BG1);
  rG.addColorStop(1, BG1);
  ctx.fillStyle = rG;
  ctx.fillRect(W - edgeWidth, 0, edgeWidth, H);

  const bG = ctx.createLinearGradient(0, H * 0.68, 0, H * 0.95);
  bG.addColorStop(0, BG0);
  bG.addColorStop(1, BG1);
  ctx.fillStyle = bG;
  ctx.fillRect(0, H * 0.68, W, H);

  const tG = ctx.createLinearGradient(0, 0, 0, H * 0.06);
  tG.addColorStop(0, BG1);
  tG.addColorStop(1, BG0);
  ctx.fillStyle = tG;
  ctx.fillRect(0, 0, W, H * 0.06);
}

export default function AuroraCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafHandleRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (canvas.dataset.auroraInit === "1") return;
    canvas.dataset.auroraInit = "1";

    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    let scrollRaf = 0;
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      drawAurora(canvas);
    };

    const onResize = () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resize, 80);
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        resize();
      }
    };

    const onPageShow = () => {
      resize();
    };

    const onScroll = () => {
      if (prefersReducedMotion) return;
      if (scrollRaf) return;
      scrollRaf = window.requestAnimationFrame(() => {
        const shift = Math.max(Math.min(window.scrollY * 0.04, 14), 0);
        canvas.style.transform = `translate3d(0, ${shift}px, 0)`;
        scrollRaf = 0;
      });
    };

    // Draw immediately so the background is available on first paint without waiting for idle.
    resize();
    // One extra frame draw helps after font/layout stabilization on first load.
    rafHandleRef.current = window.requestAnimationFrame(() => resize());
    window.addEventListener("resize", onResize);
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("pageshow", onPageShow);
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("pageshow", onPageShow);
      delete canvas.dataset.auroraInit;
      if (rafHandleRef.current !== null) {
        window.cancelAnimationFrame(rafHandleRef.current);
      }
      if (scrollRaf) window.cancelAnimationFrame(scrollRaf);
      if (resizeTimer) clearTimeout(resizeTimer);
    };
  }, []);

  return (
    <div className="aurora-scene" aria-hidden="true">
      <canvas ref={canvasRef} className="aurora-canvas" />
      <div className="vignette" />
      <div className="grid-overlay" />
      <div className="grain" />
    </div>
  );
}
