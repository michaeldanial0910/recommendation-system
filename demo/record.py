"""(Used to make demo/recommender_demo.mp4.) Drive the real Streamlit app in a visible Chromium on Xvfb and log timestamps of each scene.
Run under:  xvfb-run -s "-screen 0 1920x1080x24" python record.py  (ffmpeg x11grab started by this script)
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
URL = "http://localhost:8501"
marks = []
T0 = None

def mark(name):
    marks.append({"name": name, "t": round(time.time() - T0, 2)})
    print(f"{marks[-1]['t']:7.2f}  {name}", flush=True)

CURSOR_JS = """
(() => {
  if (window.__cursor) return;
  const c = document.createElement('div');
  c.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2 L4 19 L8.6 14.8 L11.6 21.4 L14.4 20.2 L11.4 13.6 L17.6 13.4 Z"
     fill="#1C2333" stroke="white" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex: 2147483647, pointerEvents:'none',
                          transform:'translate(-4px,-2px)', transition:'transform 60ms'});
  const ring = document.createElement('div');
  Object.assign(ring.style, {position:'fixed', width:'34px', height:'34px', borderRadius:'50%', border:'2px solid #2A6FD6',
     zIndex: 2147483646, pointerEvents:'none', opacity:'0', transform:'translate(-17px,-17px) scale(0.4)',
     transition:'opacity 350ms, transform 350ms'});
  const add = () => { document.body.appendChild(ring); document.body.appendChild(c); };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX+'px'; c.style.top = e.clientY+'px';
                                               ring.style.left = e.clientX+'px'; ring.style.top = e.clientY+'px'; }, true);
  document.addEventListener('mousedown', () => { ring.style.transition='none'; ring.style.opacity='0.9';
      ring.style.transform='translate(-17px,-17px) scale(0.4)'; requestAnimationFrame(() => { ring.style.transition='opacity 450ms, transform 450ms';
      ring.style.opacity='0'; ring.style.transform='translate(-17px,-17px) scale(1.3)'; }); }, true);
  window.__cursor = true;
})();
"""

def main():
    global T0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "30",
                           "-video_size", "1920x1080", "-draw_mouse", "0", "-i", os.environ["DISPLAY"],
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                           f"{OUT}/raw.mp4"], stdin=subprocess.PIPE)
    T0 = time.time()
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context("/tmp/rec-demo-profile", executable_path="/opt/pw-browsers/chromium",
                              headless=False, no_viewport=True,
                              args=["--kiosk", "--window-position=0,0", "--window-size=1920,1080",
                                    "--force-device-scale-factor=1.25", "--hide-scrollbars",
                                    "--disable-infobars", "--no-first-run"])
        ctx.add_init_script(CURSOR_JS)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        pg.goto(URL)
        pg.wait_for_selector("text=Ten movies for someone we already know", timeout=60000)
        pg.wait_for_selector(".ticket", timeout=60000)
        pg.wait_for_timeout(2500)          # fonts settle
        pg.evaluate(CURSOR_JS)
        m = pg.mouse
        m.move(760, 500)
        mark("ready")
        main_el = "section.stMain, [data-testid='stMain']"

        def glide(locator, steps=28, click=True, dx=0, dy=0):
            box = locator.bounding_box()
            x, y = box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy
            m.move(x, y, steps=steps)
            pg.wait_for_timeout(180)
            if click:
                m.click(x, y)

        def scroll_to(px, dur_ms=1600):
            pg.evaluate(f"""(() => {{ const el = document.querySelector("{main_el}") || document.scrollingElement;
               const start = el.scrollTop, end = {px}, t0 = performance.now(), d = {dur_ms};
               function step(t) {{ const k = Math.min(1, (t - t0) / d), e = k<.5 ? 2*k*k : 1-Math.pow(-2*k+2,2)/2;
                  el.scrollTop = start + (end - start) * e; if (k < 1) requestAnimationFrame(step); }}
               requestAnimationFrame(step); }})()""")
            pg.wait_for_timeout(dur_ms + 150)

        def settle(sel=None, ms=900):
            if sel:
                pg.wait_for_selector(sel, timeout=60000)
            pg.wait_for_timeout(ms)

        # ---- For you ----
        mark("foryou_start")
        pg.wait_for_timeout(1200)
        glide(pg.get_by_text("Comedy & family", exact=True))
        settle(ms=1700)
        mark("foryou_comedy")
        glide(pg.get_by_text("Hindsight", exact=True))
        settle("text=rated 4+ later", ms=1600)
        mark("foryou_hindsight")
        scroll_to(260, 1300)
        pg.wait_for_timeout(1300)
        mark("foryou_end")

        # ---- New user ----
        glide(pg.get_by_role("link", name="New user"))
        settle("text=Route: Popularity", 900)
        mark("new_start")
        glide(pg.get_by_text("Woman", exact=True))
        settle(ms=500)
        glide(pg.locator("[data-testid='stSelectbox']").first)
        settle(ms=500)
        glide(pg.get_by_role("option", name="25-34"))
        settle("text=Route: Popular in age/gender group", 1300)
        mark("new_demo")
        box = pg.locator("[data-testid='stMultiSelect'] input").first
        for query, option in [("Toy Story", "Toy Story (1995)"), ("Groundhog", "Groundhog Day (1993)"),
                              ("Babe (", "Babe (1995)")]:
            glide(box)
            pg.wait_for_timeout(200)
            pg.keyboard.type(query, delay=60)
            pg.wait_for_timeout(600)
            glide(pg.get_by_role("option", name=option, exact=True), steps=16)
            pg.wait_for_timeout(700)
        m.click(1700, 640)                 # close the dropdown by clicking empty page
        settle("text=Route: Hybrid + demographics", 1500)
        mark("new_hybrid")
        m.move(700, 760, steps=20)
        pg.wait_for_timeout(1400)
        mark("new_end")

        # ---- Results ----
        glide(pg.get_by_role("link", name="Results"))
        settle("text=NDCG@10 by user group", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1500)
        mark("results_start")
        scroll_to(330, 1200)
        pg.wait_for_timeout(900)
        glide(pg.locator("[data-testid='stButtonGroup'] button", has_text="New users after 3 picks"))
        settle(ms=1700)
        mark("results_onboard")
        scroll_to(1450, 1600)
        pg.wait_for_timeout(1500)
        mark("results_end")

        # ---- Why this? ----
        glide(pg.get_by_role("link", name="Why this?"))
        settle("text=Your liked movies that point to it", 1300)
        mark("why_start")
        scroll_to(380, 1300)
        pg.wait_for_timeout(500)
        glide(pg.get_by_role("button", name="Remove the 3 cited movies and re-rank"))
        settle("[data-testid='stAlert']", 1500)
        mark("why_deleted")
        scroll_to(900, 1400)
        pg.wait_for_timeout(1500)
        mark("why_end")

        # ---- The data ----
        glide(pg.get_by_role("link", name="The data"))
        settle("text=Why a time split, not a random one", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1200)
        mark("data_start")
        scroll_to(330, 1400)
        pg.wait_for_timeout(1600)
        mark("data_end")
        ctx.close()
    ff.communicate(b"q")
    json.dump(marks, open(f"{OUT}/marks.json", "w"), indent=1)

main()
