# SLOForge artifact explorer

This is a static, framework-free TypeScript UI for a generated SLOForge
`report-data.json`. It has no production data fixture, backend, authentication,
or external runtime dependency.

```bash
cd ui
npm ci
npm run dev
```

Use **Load artifact** to choose a local report. A served artifact can instead be
selected with `?data=/path/report-data.json`, or by setting
`VITE_REPORT_URL` when building. For example, after `npm run build`, serve the
repository root and open:

```text
http://127.0.0.1:4175/ui/dist/?data=/reports/demo/report-data.json
```

Validation commands:

```bash
npm run typecheck
npm run lint
npm test
npm run build
```

The generated-demo conformance test runs when
`reports/demo/report-data.json` exists and is skipped when that generated
artifact is absent. All other tests use an explicitly labeled test fixture.
