// tinyPIP-0 browser chat. Runs the trained model directly in the browser, with no server.
// Loads model/manifest.json and model/weights.bin (written by export_web.py), then
// mirrors the forward pass in model.py and the chat loop in generate.py.
"use strict";

(function () {
  var EPS = 1e-5;

  var logEl = document.getElementById("log");
  var form = document.getElementById("form");
  var input = document.getElementById("input");
  var sendBtn = document.getElementById("send");
  var statusEl = document.getElementById("status");
  var tempEl = document.getElementById("temp");
  var topkEl = document.getElementById("topk");

  var model = null;
  var history = "";

  function addLine(cls, text) {
    var d = document.createElement("div");
    d.className = cls;
    d.textContent = text;
    logEl.appendChild(d);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function tensor(name) {
    var e = model.tensors[name];
    if (!e) throw new Error("missing tensor: " + name);
    return model.weights.subarray(e.offset, e.offset + e.length);
  }

  // x: [n, C] row-major. Normalizes each row over C, then applies weight and bias.
  function layerNorm(x, n, C, w, b) {
    var out = new Float32Array(n * C);
    for (var t = 0; t < n; t++) {
      var o = t * C;
      var mean = 0;
      for (var i = 0; i < C; i++) mean += x[o + i];
      mean /= C;
      var v = 0;
      for (i = 0; i < C; i++) {
        var d = x[o + i] - mean;
        v += d * d;
      }
      v /= C;
      var inv = 1 / Math.sqrt(v + EPS);
      for (i = 0; i < C; i++) out[o + i] = (x[o + i] - mean) * inv * w[i] + b[i];
    }
    return out;
  }

  // torch nn.Linear with no bias: y[t, o] = sum_i x[t, i] * W[o, i], W shape [outDim, inDim].
  function linear(x, n, inDim, outDim, W) {
    var out = new Float32Array(n * outDim);
    for (var t = 0; t < n; t++) {
      var xo = t * inDim;
      var yo = t * outDim;
      for (var o = 0; o < outDim; o++) {
        var wo = o * inDim;
        var s = 0;
        for (var i = 0; i < inDim; i++) s += x[xo + i] * W[wo + i];
        out[yo + o] = s;
      }
    }
    return out;
  }

  // Abramowitz and Stegun 7.1.26. Maximum error about 1.5e-7.
  function erf(x) {
    var sign = x < 0 ? -1 : 1;
    var ax = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * ax);
    var poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - poly * Math.exp(-ax * ax));
  }

  // Exact GELU, matching torch F.gelu default.
  function gelu(x) {
    return 0.5 * x * (1 + erf(x / Math.SQRT2));
  }

  // Forward pass for one window of token ids. Returns the logits for the last position.
  function forward(ids) {
    var cfg = model.config;
    var C = cfg.n_embd;
    var H = cfg.n_head;
    var L = cfg.n_layer;
    var V = cfg.vocab_size;
    var n = ids.length;
    var hs = C / H;
    var scale = 1 / Math.sqrt(hs);

    var tok = tensor("tok_emb.weight");
    var pos = tensor("pos_emb.weight");
    var x = new Float32Array(n * C);
    for (var t = 0; t < n; t++) {
      for (var i = 0; i < C; i++) x[t * C + i] = tok[ids[t] * C + i] + pos[t * C + i];
    }

    for (var l = 0; l < L; l++) {
      var p = "blocks." + l + ".";

      // causal self-attention
      var h1 = layerNorm(x, n, C, tensor(p + "ln1.weight"), tensor(p + "ln1.bias"));
      var qkv = linear(h1, n, C, 3 * C, tensor(p + "attn.qkv.weight"));
      var att = new Float32Array(n * C);
      var sc = new Float64Array(n);
      for (var h = 0; h < H; h++) {
        var qo = h * hs;
        var ko = C + h * hs;
        var vo = 2 * C + h * hs;
        for (t = 0; t < n; t++) {
          var qb = t * 3 * C + qo;
          var max = -Infinity;
          for (var j = 0; j <= t; j++) {
            var kb = j * 3 * C + ko;
            var s = 0;
            for (var d = 0; d < hs; d++) s += qkv[qb + d] * qkv[kb + d];
            s *= scale;
            sc[j] = s;
            if (s > max) max = s;
          }
          var sum = 0;
          for (j = 0; j <= t; j++) {
            sc[j] = Math.exp(sc[j] - max);
            sum += sc[j];
          }
          for (d = 0; d < hs; d++) {
            var acc = 0;
            for (j = 0; j <= t; j++) acc += sc[j] * qkv[j * 3 * C + vo + d];
            att[t * C + h * hs + d] = acc / sum;
          }
        }
      }
      var proj = linear(att, n, C, C, tensor(p + "attn.proj.weight"));
      for (i = 0; i < n * C; i++) x[i] += proj[i];

      // MLP: fc (C to 4C), GELU, proj (4C to C)
      var h2 = layerNorm(x, n, C, tensor(p + "ln2.weight"), tensor(p + "ln2.bias"));
      var fc = linear(h2, n, C, 4 * C, tensor(p + "mlp.fc.weight"));
      for (i = 0; i < fc.length; i++) fc[i] = gelu(fc[i]);
      var mp = linear(fc, n, 4 * C, C, tensor(p + "mlp.proj.weight"));
      for (i = 0; i < n * C; i++) x[i] += mp[i];
    }

    var hf = layerNorm(x, n, C, tensor("ln_f.weight"), tensor("ln_f.bias"));
    var last = hf.subarray((n - 1) * C, n * C);
    // output head is tied to the token embedding, so the logits use tok as the weight
    return linear(last, 1, C, V, tok);
  }

  // Temperature and top-k sampling, matching model.generate in model.py.
  function sample(logits, temperature, topK) {
    var V = logits.length;
    var t = Math.max(temperature, 1e-5);
    var z = new Float64Array(V);
    var i;
    for (i = 0; i < V; i++) z[i] = logits[i] / t;
    if (topK > 0 && topK < V) {
      var sorted = Array.from(z).sort(function (a, b) { return b - a; });
      var thr = sorted[topK - 1];
      for (i = 0; i < V; i++) if (z[i] < thr) z[i] = -Infinity;
    }
    var max = -Infinity;
    for (i = 0; i < V; i++) if (z[i] > max) max = z[i];
    var probs = new Float64Array(V);
    var total = 0;
    for (i = 0; i < V; i++) {
      probs[i] = Math.exp(z[i] - max);
      total += probs[i];
    }
    var r = Math.random() * total;
    var acc = 0;
    for (i = 0; i < V; i++) {
      acc += probs[i];
      if (r < acc) return i;
    }
    return V - 1;
  }

  // Generates one reply. Stops at a newline, like generate.py's chat mode.
  function generate(promptText, maxTokens, temperature, topK) {
    var block = model.config.block_size;
    var ids = Array.from(promptText).map(function (ch) { return model.stoi.get(ch); });
    var out = [];
    for (var step = 0; step < maxTokens; step++) {
      var window = ids.slice(-block);
      var logits = forward(window);
      var next = sample(logits, temperature, topK);
      ids.push(next);
      var ch = model.itos.get(next);
      if (ch === "\n") break;
      out.push(ch);
    }
    return out.join("").replace(/^ +/, "").split("\n")[0].trim();
  }

  async function load() {
    var manifestRes = await fetch("model/manifest.json");
    if (!manifestRes.ok) {
      throw new Error("model/manifest.json not found (HTTP " + manifestRes.status + "). Run a training workflow first.");
    }
    var manifest = await manifestRes.json();
    var weightsRes = await fetch("model/weights.bin");
    if (!weightsRes.ok) throw new Error("model/weights.bin not found (HTTP " + weightsRes.status + ")");
    var weights = new Float32Array(await weightsRes.arrayBuffer());
    if (weights.length !== manifest.total_floats) {
      throw new Error("weights.bin has " + weights.length + " floats but manifest expects " + manifest.total_floats);
    }
    var itos = new Map();
    var stoi = new Map();
    Object.keys(manifest.itos).forEach(function (k) {
      var ch = manifest.itos[k];
      itos.set(Number(k), ch);
      stoi.set(ch, Number(k));
    });
    model = { config: manifest.config, tensors: manifest.tensors, weights: weights, itos: itos, stoi: stoi };
    sendBtn.disabled = false;
    setStatus("ready. " + weights.length.toLocaleString("en-US") + " parameters loaded (training step " + manifest.step + "). Say something.");
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    var text = input.value.trim();
    input.value = "";
    if (!text) return;
    if (text === "/reset") {
      history = "";
      addLine("sys", "(history cleared)");
      return;
    }
    if (!model) {
      addLine("sys", "model is not loaded yet");
      return;
    }

    var turn = "User: " + text + "\nPip:";
    var unknown = [];
    Array.from(turn).forEach(function (ch) {
      if (!model.stoi.has(ch) && unknown.indexOf(ch) < 0) unknown.push(ch);
    });
    if (unknown.length) {
      addLine("sys", "(skipped: message has characters not in vocabulary: " + JSON.stringify(unknown) + ")");
      return;
    }

    addLine("you", "You: " + text);
    sendBtn.disabled = true;
    setStatus("thinking...");
    await new Promise(function (r) { setTimeout(r, 20); });
    try {
      history += turn;
      var temperature = Number(tempEl.value) || 0.8;
      var topK = Number(topkEl.value) || 0;
      var reply = generate(history, 300, temperature, topK);
      history += " " + reply + "\n";
      addLine("pip", "Pip: " + (reply || "(no reply)"));
    } catch (err) {
      addLine("sys", "error: " + err.message);
    } finally {
      sendBtn.disabled = false;
      setStatus("ready");
      input.focus();
    }
  });

  load().catch(function (err) {
    setStatus("could not load the model: " + err.message);
  });
})();
