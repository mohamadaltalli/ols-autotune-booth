# Ol's Autotune Booth

Records a take in the browser, finds the pitch with YIN, and re-tunes it with
TD-PSOLA so it snaps to a musical scale. All the audio work happens on the
device — nothing is uploaded.

The microphone is captured as raw PCM straight off the Web Audio graph, rather
than through `MediaRecorder` and back out of `decodeAudioData`. There is no
codec in the path, so there is nothing for a platform to disagree about — which
is what used to break the whole booth on iOS Safari.

Sign-in is `ol` / `ray13`. Signing in as `talli` / `nebraska` opens the same
booth with a diagnostics panel available from the header — audio context state,
audio session routing, captured sample count, voiced-frame ratio, per-stage
timings and the last error. It exists because a phone has no console.

Both credential pairs are in the bundle and readable by anyone who opens
devtools; the gate is for presentation, not security.

## Run it locally

```bash
npm install
npm run dev
```

Open the printed localhost URL. The microphone works on `localhost` even
over plain HTTP, because browsers treat it as a secure origin.

## Publish it

1. Create a repository on GitHub and push this folder to the `main` branch.
2. In the repo, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

That's it. Every push to `main` builds and deploys. The first run takes a
couple of minutes; the URL appears in the workflow summary and on the Pages
settings screen.

### How the base path gets set

Pages serves a project site from `/<repo>/`, so asset URLs need that prefix.
Getting it wrong is the usual cause of a Pages site loading a blank page with
404s in the console.

You don't need to set it by hand. The workflow passes the repo name to the
build step:

```yaml
- name: Build
  run: npm run build
  env:
    PAGES_BASE: ${{ github.repository }}
```

and `vite.config.js` turns that into the base path. It falls back to
`GITHUB_REPOSITORY` — a variable GitHub sets on every runner automatically — so
the build still resolves correctly even if that `env` line is removed. Locally
neither is set and the base stays `/`.

Every build prints the base it used, so you can check it in the Actions log:

```
[vite] building with base "/ols-autotune-booth/"
```

## Microphone requirements

`getUserMedia` only runs in a secure context. GitHub Pages is HTTPS, so it
works there. It will *not* work if you open `dist/index.html` directly as a
`file://` URL — use `npm run preview` to check a production build.

## Custom domain

Add a `public/CNAME` file containing your domain, and set the domain under
**Settings → Pages**. `vite.config.js` will still emit a `/<repo>/` base, which
is wrong for a custom domain, so change the last line of that file to:

```js
export default defineConfig({ base: '/', plugins: [react()] })
```

## Layout

```
index.html                    entry document
src/main.jsx                  mounts React
src/OlsAutotuneBooth.jsx    the app, DSP engine included
vite.config.js                base path logic
.github/workflows/deploy.yml  build and deploy on push to main
```
