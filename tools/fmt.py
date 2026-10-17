import json, sys
for l in sys.stdin:
    try:
        r = json.loads(l)
    except Exception:
        print(l.strip()); continue
    a = r["audio_s"]
    print(f"audio {a:.1f}s | llm {r['tok_per_s']:.0f} tok/s | decoder {r['decoder_s']/a:.2f}x upsampler {r['upsampler_s']/a:.2f}x"
          f" | total {r['total_s']:.1f}s (RTF {r['total_s']/a:.2f}) | first audio {r['first_audio_s']:.1f}s | stalls {r.get('stall_s', -1):.1f}s"
          f" | llm thr {r['threads']} onnx thr {r.get('onnx_threads')} hd {r['hd']} xnn {r['xnnpack']} seed {r['seed']}")
