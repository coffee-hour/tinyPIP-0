# web/ (talk with Pip in the browser)

This folder is a static site. It is the Cloudflare Pages output directory for coffee-hour/tinyPIP-0.

Files:
- index.html: the chat page.
- app.js: runs the model in the browser (a JavaScript forward pass that mirrors model.py).
- model/: weights.bin and manifest.json, written by export_web.py during a training run. They do not exist until the first run finishes. Until then the page says the model could not be loaded.

## Cloudflare Pages setup (one time, in the Cloudflare dashboard)

1. Workers & Pages, then Create, then Pages, then Connect to Git.
2. Pick the coffee-hour/tinyPIP-0 repository and the main branch.
3. Build command: leave empty.
4. Build output directory: web
5. Save and deploy.

The site is served at the project's pages.dev address.

## Updating the model

Run the Train tinyPIP-0 workflow on GitHub Actions. When it succeeds it commits out/tinypip0.pt and web/model/ to main. Cloudflare Pages should then rebuild and deploy the new weights automatically. If commits marked [skip ci] do not trigger a Pages build, start a deployment from the Cloudflare dashboard.

To export a checkpoint by hand:

    python export_web.py --ckpt out/tinypip0.pt --out web/model
