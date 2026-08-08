# Ol's Autotune Booth

Records a take in the browser, finds the pitch with YIN, and re-tunes it with
TD-PSOLA so it snaps to a musical scale. All the audio work happens on the
device — nothing is uploaded.

Sign-in is `midnight` / `ray13`. Those credentials are in the bundle and
readable by anyone who opens devtools; the gate is for presentation, not
security.

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

You do **not** need to set the `base` path by hand. `vite.config.js` reads the
repository name from the environment during the Actions run and prefixes asset
URLs with it. Getting this wrong is the usual cause of a Pages site loading a
blank page with 404s in the console.

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
