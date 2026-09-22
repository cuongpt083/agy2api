import { useEffect, useRef, useState } from 'react';
import { useApiKey } from '../hooks/use-api-key';
import { fileToBase64 } from '../lib/utils';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Image as ImageIcon, Loader2, X } from 'lucide-react';

const SIZE_OPTIONS = [
  { value: '1024x1024', label: '1024×1024' },
  { value: '1792x1024', label: '1792×1024' },
  { value: '1024x1792', label: '1024×1792' },
  { value: '1:1', label: '1:1' },
  { value: '16:9', label: '16:9' },
  { value: '9:16', label: '9:16' },
];

function fieldClass() {
  return 'h-8 w-full rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50';
}

export function ImagesPage() {
  const { apiKey } = useApiKey();
  const [prompt, setPrompt] = useState('');
  const [size, setSize] = useState('1024x1024');
  const [n, setN] = useState(1);
  const [model, setModel] = useState('');
  const [models, setModels] = useState<string[]>([]);
  const [references, setReferences] = useState<{ file: File; url: string }[]>([]);
  const [images, setImages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isLoading) return;
    setElapsed(0);
    const id = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [isLoading]);

  useEffect(() => {
    if (!apiKey) return;
    fetch('/v1/models', { headers: { Authorization: `Bearer ${apiKey}` } })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.data) {
          setModels(data.data.map((m: { id: string }) => m.id).filter(Boolean));
        }
      })
      .catch(() => {});
  }, [apiKey]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    const incoming = Array.from(e.target.files).slice(0, 3 - references.length);
    setReferences((prev) => [
      ...prev,
      ...incoming.map((file) => ({ file, url: URL.createObjectURL(file) })),
    ]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    if (!apiKey) {
      alert('Please set your API key in the Keys page first.');
      return;
    }

    setIsLoading(true);
    setStatus('started');
    setError(null);

    try {
      const reference_images = await Promise.all(references.map((r) => fileToBase64(r.file)));
      const body: Record<string, unknown> = {
        prompt: prompt.trim(),
        n,
        size,
        response_format: 'url',
        stream: true,
      };
      if (model.trim()) body.model = model.trim();
      if (reference_images.length) body.reference_images = reference_images;

      const response = await fetch('/v1/images/generations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.error?.message || `API Error: ${response.status} ${response.statusText}`);
      }
      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') continue;
          const ev = JSON.parse(payload);
          if (ev.type === 'status' && ev.stage) {
            setStatus(ev.stage);
          } else if (ev.type === 'image' && Array.isArray(ev.data)) {
            const newImages = ev.data.map(
              (img: { url?: string; b64_json?: string }) =>
                img.url || `data:image/png;base64,${img.b64_json}`,
            );
            setImages((prev) => [...newImages, ...prev]);
          } else if (ev.type === 'error') {
            throw new Error(ev.error?.message || 'Image generation failed');
          }
        }
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
      setStatus(null);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background p-8 overflow-auto">
      <div className="max-w-4xl mx-auto w-full space-y-8">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Generate Images</h1>
          <p className="text-sm text-muted-foreground mt-2">
            POST /v1/images/generations — text-to-image or image-to-image via AGY.
          </p>
        </div>

        <form onSubmit={handleGenerate} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Prompt</label>
            <Input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the image you want to generate..."
              disabled={isLoading}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Size</label>
              <select
                className={fieldClass()}
                value={size}
                onChange={(e) => setSize(e.target.value)}
                disabled={isLoading}
              >
                {SIZE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Count (n)</label>
              <select
                className={fieldClass()}
                value={n}
                onChange={(e) => setN(Number(e.target.value))}
                disabled={isLoading}
              >
                {[1, 2, 3, 4].map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Model (optional)</label>
              <Input
                list="agy-models"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="Default AGY model"
                disabled={isLoading}
              />
              <datalist id="agy-models">
                {models.map((id) => (
                  <option key={id} value={id} />
                ))}
              </datalist>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Reference images (optional, max 3)</label>
            <div className="flex flex-wrap gap-3 items-center">
              {references.map((ref, i) => (
                <div key={ref.url} className="relative w-16 h-16 rounded-md overflow-hidden border">
                  <img src={ref.url} alt="" className="w-full h-full object-cover" />
                  <button
                    type="button"
                    className="absolute top-0.5 right-0.5 bg-black/60 text-white rounded-full p-0.5"
                    onClick={() => setReferences((prev) => prev.filter((_, idx) => idx !== i))}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
              {references.length < 3 && (
                <Button
                  type="button"
                  variant="outline"
                  disabled={isLoading}
                  onClick={() => fileInputRef.current?.click()}
                >
                  Add image
                </Button>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button type="submit" disabled={isLoading || !prompt.trim()} className="gap-2">
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ImageIcon className="w-4 h-4" />}
              Generate
            </Button>
            {isLoading && (
              <p className="text-sm text-muted-foreground">
                {status === 'generating' ? 'Generating' : 'Starting'} with AGY… {elapsed}s
              </p>
            )}
          </div>
        </form>

        {error && (
          <div className="p-4 bg-destructive/10 text-destructive rounded-xl text-sm">{error}</div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          {images.map((img, i) => (
            <div
              key={i}
              className="border rounded-2xl overflow-hidden bg-card shadow-sm group relative flex justify-center bg-black/5"
            >
              <img src={img} alt="Generated" className="w-full h-auto object-contain" />
            </div>
          ))}
          {images.length === 0 && !isLoading && !error && (
            <div className="col-span-full py-20 text-center text-muted-foreground border-2 border-dashed rounded-2xl">
              No images generated yet.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
