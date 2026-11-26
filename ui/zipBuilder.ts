/**
 * Pure TypeScript helpers for saiverse-voice-tts PersonaPanel:
 *   1. Batch input parsing (JSON array or line-by-line text)
 *   2. Safe WAV filename sanitization & deduplication
 *   3. Dependency-free ZIP builder (STORE + CRC-32 + UTF-8 flag 0x0800)
 *
 * Kept pure with zero DOM dependencies so Node unit tests can exercise
 * every edge case directly.
 */

export type BatchInputMode = "json" | "lines";

export interface ParsedBatchItem {
    index: number;
    text: string;
    filename: string | null;
    personaId: string | null;
}

export interface BatchParseResult {
    items: ParsedBatchItem[];
    error: string | null;
}

export interface SingleSynthCacheKey {
    personaId: string;
    text: string;
    applyPronunciation: boolean;
    configRevision: number | null;
}

/**
 * Decide whether a previously synthesized preview WAV can be reused for saving.
 *
 * Requires a non-null `configRevision` from the host so that any edit to
 * persona/global voice settings (Voice ID, engine, reference audio, speed,
 * pronunciation dictionary, etc.) invalidates the cached audio. When the host
 * does not supply a revision (`null`), fail closed and force re-synthesis.
 */
export function canReuseSingleSynthCache(
    cached: SingleSynthCacheKey | null | undefined,
    current: SingleSynthCacheKey,
): boolean {
    if (!cached) return false;
    if (cached.configRevision === null || current.configRevision === null) {
        return false;
    }
    return (
        cached.configRevision === current.configRevision &&
        cached.personaId === current.personaId &&
        cached.text === current.text &&
        cached.applyPronunciation === current.applyPronunciation
    );
}

export const BATCH_MAX_ITEMS = 200;

const INVALID_FILENAME_CHARS_RE = /[<>:"/\\|?*\x00-\x1f\x7f]/g;
const WINDOWS_RESERVED_STEMS = new Set([
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
]);

/**
 * Sanitize a user-supplied filename into a safe single-segment `.wav` filename.
 * Mirrors backend `api_routes._sanitize_wav_filename`.
 */
export function sanitizeWavFilename(raw: string | null | undefined, fallbackStem: string): string {
    let text = String(raw ?? "").trim();
    if (text) {
        text = text.replace(/\//g, "_").replace(/\\/g, "_");
        text = text.replace(INVALID_FILENAME_CHARS_RE, "_");
        text = text.replace(/\.\./g, "_");
        text = text.replace(/_+/g, "_");
        text = text.replace(/^[ ._]+|[ ._]+$/g, "");
    }
    if (text.toLowerCase().endsWith(".wav")) {
        text = text.slice(0, -4).replace(/^[ ._]+|[ ._]+$/g, "");
    }
    if (!text) {
        text = fallbackStem;
    }
    if (text.length > 100) {
        text = text.slice(0, 100).replace(/[ ._]+$/g, "") || fallbackStem;
    }
    if (WINDOWS_RESERVED_STEMS.has(text.toUpperCase())) {
        text = `${text}_`;
    }
    return `${text}.wav`;
}

export function allocateUniqueWavName(candidate: string, usedLower: Set<string>): string {
    const lower = candidate.toLowerCase();
    if (!usedLower.has(lower)) {
        usedLower.add(lower);
        return candidate;
    }
    const stem = lower.endsWith(".wav") ? candidate.slice(0, -4) : candidate;
    let suffix = 2;
    while (true) {
        const alt = `${stem}_${suffix}.wav`;
        if (!usedLower.has(alt.toLowerCase())) {
            usedLower.add(alt.toLowerCase());
            return alt;
        }
        suffix += 1;
    }
}

export function parseBatchInput(raw: string, mode: BatchInputMode): BatchParseResult {
    const trimmed = raw.trim();
    if (!trimmed) {
        return { items: [], error: null };
    }

    if (mode === "lines") {
        const lines = raw
            .replace(/\r\n/g, "\n")
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line.length > 0);
        if (lines.length > BATCH_MAX_ITEMS) {
            return {
                items: [],
                error: `件数が上限 (${BATCH_MAX_ITEMS} 件) を超えています (${lines.length} 件)。分割して実行してください。`,
            };
        }
        return {
            items: lines.map((text, idx) => ({
                index: idx + 1,
                text,
                filename: null,
                personaId: null,
            })),
            error: null,
        };
    }

    // JSON array mode
    let parsed: unknown;
    try {
        parsed = JSON.parse(trimmed);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return { items: [], error: `JSON の構文エラー: ${msg}` };
    }

    if (!Array.isArray(parsed)) {
        return {
            items: [],
            error: 'JSON は配列形式で指定してください (例: ["おはよう！", {"filename": "win", "text": "やったね！"}])。',
        };
    }

    if (parsed.length === 0) {
        return { items: [], error: "配列が空です。" };
    }

    if (parsed.length > BATCH_MAX_ITEMS) {
        return {
            items: [],
            error: `件数が上限 (${BATCH_MAX_ITEMS} 件) を超えています (${parsed.length} 件)。分割して実行してください。`,
        };
    }

    const items: ParsedBatchItem[] = [];
    for (let i = 0; i < parsed.length; i++) {
        const entry = parsed[i];
        if (typeof entry === "string") {
            const text = entry.trim();
            if (!text) {
                return { items: [], error: `${i + 1} 番目のテキストが空です。` };
            }
            items.push({ index: i + 1, text, filename: null, personaId: null });
        } else if (entry && typeof entry === "object" && !Array.isArray(entry)) {
            const obj = entry as Record<string, unknown>;
            if (typeof obj.text !== "string" || !obj.text.trim()) {
                return {
                    items: [],
                    error: `${i + 1} 番目の要素には空でない文字列の "text" が必要です。`,
                };
            }
            const filename = obj.filename !== undefined && obj.filename !== null
                ? String(obj.filename).trim() || null
                : null;
            const personaId = obj.persona_id !== undefined && obj.persona_id !== null
                ? String(obj.persona_id).trim() || null
                : null;
            items.push({
                index: i + 1,
                text: obj.text.trim(),
                filename,
                personaId,
            });
        } else {
            return {
                items: [],
                error: `${i + 1} 番目の要素は文字列、または {"text": "...", "filename": "..."} 形式のオブジェクトにしてください。`,
            };
        }
    }

    return { items, error: null };
}

// ---------------------------------------------------------------------------
// Minimal standards-compliant ZIP archive builder (STORE, UTF-8 names, CRC-32)
// ---------------------------------------------------------------------------

export interface ZipEntryInput {
    name: string;
    data: Uint8Array;
}

const CRC32_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) {
            c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        }
        table[i] = c >>> 0;
    }
    return table;
})();

export function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
        crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function dosTimeDate(now: Date): { dosTime: number; dosDate: number } {
    const year = Math.max(1980, now.getFullYear());
    const dosTime =
        ((now.getHours() & 0x1f) << 11) |
        ((now.getMinutes() & 0x3f) << 5) |
        ((now.getSeconds() >> 1) & 0x1f);
    const dosDate =
        (((year - 1980) & 0x7f) << 9) |
        (((now.getMonth() + 1) & 0x0f) << 5) |
        (now.getDate() & 0x1f);
    return { dosTime, dosDate };
}

/**
 * Build a valid ZIP archive byte array (method 0 / STORE, UTF-8 filename flag).
 */
export function buildZipBytes(entries: ZipEntryInput[], now: Date = new Date()): Uint8Array {
    const encoder = new TextEncoder();
    const { dosTime, dosDate } = dosTimeDate(now);

    const prepared = entries.map((e) => {
        const nameBytes = encoder.encode(e.name);
        const crc = crc32(e.data);
        return { nameBytes, data: e.data, crc, offset: 0 };
    });

    let totalLocalSize = 0;
    let totalCentralSize = 0;
    for (const p of prepared) {
        totalLocalSize += 30 + p.nameBytes.length + p.data.length;
        totalCentralSize += 46 + p.nameBytes.length;
    }
    const endOfCentralDirSize = 22;
    const out = new Uint8Array(totalLocalSize + totalCentralSize + endOfCentralDirSize);
    const view = new DataView(out.buffer);

    let cursor = 0;
    for (const p of prepared) {
        p.offset = cursor;
        // Local file header signature
        view.setUint32(cursor, 0x04034b50, true);
        // Version needed to extract (2.0)
        view.setUint16(cursor + 4, 20, true);
        // General purpose bit flag: bit 11 = UTF-8 filename encoding
        view.setUint16(cursor + 6, 0x0800, true);
        // Compression method: 0 = STORE
        view.setUint16(cursor + 8, 0, true);
        view.setUint16(cursor + 10, dosTime, true);
        view.setUint16(cursor + 12, dosDate, true);
        view.setUint32(cursor + 14, p.crc, true);
        view.setUint32(cursor + 18, p.data.length, true);
        view.setUint32(cursor + 22, p.data.length, true);
        view.setUint16(cursor + 26, p.nameBytes.length, true);
        view.setUint16(cursor + 28, 0, true);
        cursor += 30;
        out.set(p.nameBytes, cursor);
        cursor += p.nameBytes.length;
        out.set(p.data, cursor);
        cursor += p.data.length;
    }

    const centralStart = cursor;
    for (const p of prepared) {
        // Central directory file header signature
        view.setUint32(cursor, 0x02014b50, true);
        view.setUint16(cursor + 4, 20, true);
        view.setUint16(cursor + 6, 20, true);
        view.setUint16(cursor + 8, 0x0800, true);
        view.setUint16(cursor + 10, 0, true);
        view.setUint16(cursor + 12, dosTime, true);
        view.setUint16(cursor + 14, dosDate, true);
        view.setUint32(cursor + 16, p.crc, true);
        view.setUint32(cursor + 20, p.data.length, true);
        view.setUint32(cursor + 24, p.data.length, true);
        view.setUint16(cursor + 28, p.nameBytes.length, true);
        view.setUint16(cursor + 30, 0, true);
        view.setUint16(cursor + 32, 0, true);
        view.setUint16(cursor + 34, 0, true);
        view.setUint16(cursor + 36, 0, true);
        view.setUint32(cursor + 38, 0, true);
        view.setUint32(cursor + 42, p.offset, true);
        cursor += 46;
        out.set(p.nameBytes, cursor);
        cursor += p.nameBytes.length;
    }

    // End of central directory record
    view.setUint32(cursor, 0x06054b50, true);
    view.setUint16(cursor + 4, 0, true);
    view.setUint16(cursor + 6, 0, true);
    view.setUint16(cursor + 8, prepared.length, true);
    view.setUint16(cursor + 10, prepared.length, true);
    view.setUint32(cursor + 12, totalCentralSize, true);
    view.setUint32(cursor + 16, centralStart, true);
    view.setUint16(cursor + 20, 0, true);

    return out;
}
