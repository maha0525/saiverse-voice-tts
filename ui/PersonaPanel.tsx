"use client";

/**
 * Voice TTS Addon - ペルソナ別設定内の音声テスト・ファイル出力パネル。
 *
 * frontend/scripts/sync-addon-panels.mjs により
 * frontend/src/addon-panels/saiverse-voice-tts/PersonaPanel.tsx へコピーされ、
 * AddonManagerModal の「ペルソナ別設定」ブロック内 (ボイス ID や参照音声のすぐ下)
 * に動的表示される。
 *
 * 機能:
 *   - 単発テスト: 選択中ペルソナの声でテキストをテスト再生 / WAV 保存
 *   - 一括 ZIP 出力: JSON 配列 (文字列配列 / {filename, text, persona_id?})
 *     または 1 行 1 セリフのテキスト (ファイル読み込み可) から順次合成し、
 *     進捗表示・途中中止つきで WAV + manifest.json の ZIP を保存
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import "./theme.module.css";
import {
    allocateUniqueWavName,
    buildZipBytes,
    canReuseSingleSynthCache,
    parseBatchInput,
    sanitizeWavFilename,
    type BatchInputMode,
} from "./zipBuilder";

interface AddonPersonaPanelProps {
    addon: {
        addon_name: string;
        display_name: string;
        version: string;
        description?: string;
    };
    personaId: string;
    personaName: string;
    personas: { id: string; name: string }[];
    addonApiBase: string;
    onConfigChanged?: () => void | Promise<void>;
    flushPendingSaves?: () => Promise<void>;
    configRevision?: number;
    getConfigRevision?: () => number;
}

type PanelTab = "single" | "batch";

interface SingleSynthCache {
    personaId: string;
    text: string;
    applyPronunciation: boolean;
    configRevision: number | null;
    blob: Blob;
    objectUrl: string;
    engine: string;
    durationSec: number | null;
    sampleRate: number | null;
    appliedText: string;
}

interface BatchResultItem {
    index: number;
    status: "ok" | "error";
    personaId: string;
    filename: string | null;
    text: string;
    appliedText?: string;
    engine?: string;
    durationSec?: number | null;
    sampleRate?: number | null;
    wavBytes?: Uint8Array;
    objectUrl?: string;
    error?: string;
}

const SAMPLE_JSON = JSON.stringify(
    [
        { filename: "greeting", text: "おはよう！今日もいい一日にしようね。" },
        { filename: "english_sample", text: "Hello, world! Ready when you are." },
        "お疲れさま、少し休憩しようか。",
    ],
    null,
    2,
);

const SAMPLE_LINES = [
    "おはよう！今日もいい一日にしようね。",
    "Hello, world! Ready when you are.",
    "お疲れさま、少し休憩しようか。",
].join("\n");

function formatTimestamp(d: Date = new Date()): string {
    const pad = (n: number) => String(n).padStart(2, "0");
    return (
        `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
        `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
    );
}

function triggerDownload(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

async function extractErrorDetail(res: Response): Promise<string> {
    const raw = await res.text().catch(() => "");
    if (!raw) return `HTTP ${res.status} ${res.statusText}`;
    try {
        const parsed = JSON.parse(raw);
        if (typeof parsed?.detail === "string" && parsed.detail) {
            return parsed.detail;
        }
    } catch {
        // plain text response
    }
    return raw.length > 300 ? `${raw.slice(0, 300)}…` : raw;
}

export default function VoiceTtsPersonaPanel({
    personaId,
    personaName,
    addonApiBase,
    flushPendingSaves,
    configRevision,
    getConfigRevision,
}: AddonPersonaPanelProps) {
    const [tab, setTab] = useState<PanelTab>("single");
    const [applyPronunciation, setApplyPronunciation] = useState<boolean>(true);

    // ----- Single test state -----
    const [singleText, setSingleText] = useState<string>("");
    const [singleFilename, setSingleFilename] = useState<string>("");
    const [singleBusy, setSingleBusy] = useState<boolean>(false);
    const [singleError, setSingleError] = useState<string | null>(null);
    const [singleResult, setSingleResult] = useState<SingleSynthCache | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // ----- Batch export state -----
    const [batchMode, setBatchMode] = useState<BatchInputMode>("json");
    const [batchInput, setBatchInput] = useState<string>("");
    const [batchBusy, setBatchBusy] = useState<boolean>(false);
    const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
    const [batchError, setBatchError] = useState<string | null>(null);
    const [batchResults, setBatchResults] = useState<BatchResultItem[]>([]);
    const [batchCancelled, setBatchCancelled] = useState<boolean>(false);
    const [previewingIndex, setPreviewingIndex] = useState<number | null>(null);
    const batchAbortRef = useRef<AbortController | null>(null);
    const batchFileInputRef = useRef<HTMLInputElement | null>(null);
    const batchAudioRef = useRef<HTMLAudioElement | null>(null);

    // Track object URLs for reliable cleanup
    const singleUrlRef = useRef<string | null>(null);
    const batchUrlsRef = useRef<string[]>([]);

    const revokeSingleResult = useCallback(() => {
        if (singleUrlRef.current) {
            URL.revokeObjectURL(singleUrlRef.current);
            singleUrlRef.current = null;
        }
        setSingleResult(null);
    }, []);

    const revokeBatchResults = useCallback(() => {
        for (const u of batchUrlsRef.current) {
            URL.revokeObjectURL(u);
        }
        batchUrlsRef.current = [];
        setBatchResults([]);
        setPreviewingIndex(null);
    }, []);

    // ペルソナ切替時: 入力文は保持して別ペルソナとの聴き比べをしやすくし、
    // 旧ペルソナの合成済み音声プレビュー・エラーだけクリアする。
    useEffect(() => {
        revokeSingleResult();
        setSingleError(null);
    }, [personaId, revokeSingleResult]);

    // Unmount cleanup
    useEffect(() => {
        return () => {
            batchAbortRef.current?.abort();
            if (singleUrlRef.current) {
                URL.revokeObjectURL(singleUrlRef.current);
                singleUrlRef.current = null;
            }
            for (const u of batchUrlsRef.current) {
                URL.revokeObjectURL(u);
            }
            batchUrlsRef.current = [];
        };
    }, []);

    // ----- Single synthesis -----
    const runSingleSynthesis = useCallback(async (
        options: { autoPlay: boolean; download: boolean },
    ) => {
        const trimmed = singleText.trim();
        if (!trimmed) {
            setSingleError("読み上げるテキストを入力してください。");
            return;
        }
        setSingleError(null);
        setSingleBusy(true);
        try {
            // 直前に編集された Voice ID・参照音声・話速・読み方辞書などの
            // 保存リクエスト完了を、キャッシュ判定より先に必ず待つ。
            await flushPendingSaves?.();
            const currentRevision = getConfigRevision?.() ?? configRevision ?? null;

            // 同じペルソナ・同じ文・同じ辞書設定・かつ設定リビジョンが変わって
            // いないときだけプレビュー済み WAV を再利用する (リビジョン未提供時は
            // 安全側で必ず再合成)。
            if (
                options.download &&
                !options.autoPlay &&
                singleResult &&
                canReuseSingleSynthCache(singleResult, {
                    personaId,
                    text: trimmed,
                    applyPronunciation,
                    configRevision: currentRevision,
                })
            ) {
                const fallback = `${personaName || personaId}_${formatTimestamp()}`;
                const saveName = sanitizeWavFilename(singleFilename, fallback);
                triggerDownload(singleResult.blob, saveName);
                return;
            }

            const res = await fetch(`${addonApiBase}/synthesize`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    persona_id: personaId,
                    text: trimmed,
                    apply_pronunciation: applyPronunciation,
                    clean_text: true,
                    filename: singleFilename.trim() || undefined,
                }),
            });
            if (!res.ok) {
                throw new Error(await extractErrorDetail(res));
            }
            const blob = await res.blob();
            const engine = res.headers.get("X-TTS-Engine") ?? "";
            const durRaw = res.headers.get("X-TTS-Duration-Sec");
            const srRaw = res.headers.get("X-TTS-Sample-Rate");
            const appliedRaw = res.headers.get("X-TTS-Applied-Text") ?? "";
            let appliedText = trimmed;
            if (appliedRaw) {
                try {
                    appliedText = decodeURIComponent(appliedRaw);
                } catch {
                    appliedText = appliedRaw;
                }
            }

            if (singleUrlRef.current) {
                URL.revokeObjectURL(singleUrlRef.current);
            }
            const objectUrl = URL.createObjectURL(blob);
            singleUrlRef.current = objectUrl;

            const cache: SingleSynthCache = {
                personaId,
                text: trimmed,
                applyPronunciation,
                configRevision: currentRevision,
                blob,
                objectUrl,
                engine,
                durationSec: durRaw ? Number(durRaw) : null,
                sampleRate: srRaw ? Number(srRaw) : null,
                appliedText,
            };
            setSingleResult(cache);

            if (options.autoPlay) {
                // DOM の <audio src={objectUrl}> 反映後に再生を開始
                window.setTimeout(() => {
                    const el = audioRef.current;
                    if (!el) return;
                    el.currentTime = 0;
                    void el.play().catch(() => {
                        // ブラウザの自動再生ポリシーで止められた場合はコントロールから再生できる
                    });
                }, 0);
            }

            if (options.download) {
                const fallback = `${personaName || personaId}_${formatTimestamp()}`;
                const saveName = sanitizeWavFilename(singleFilename, fallback);
                triggerDownload(blob, saveName);
            }
        } catch (err) {
            setSingleError(err instanceof Error ? err.message : String(err));
        } finally {
            setSingleBusy(false);
        }
    }, [
        addonApiBase,
        applyPronunciation,
        configRevision,
        flushPendingSaves,
        getConfigRevision,
        personaId,
        personaName,
        singleFilename,
        singleResult,
        singleText,
    ]);

    const isSingleResultStale = useMemo(() => {
        if (!singleResult) return false;
        const latestRev = getConfigRevision?.() ?? configRevision ?? null;
        return !canReuseSingleSynthCache(singleResult, {
            personaId,
            text: singleText.trim(),
            applyPronunciation,
            configRevision: latestRev,
        });
    }, [
        applyPronunciation,
        configRevision,
        getConfigRevision,
        personaId,
        singleResult,
        singleText,
    ]);

    // ----- Batch parsing preview -----
    const parsedBatch = useMemo(
        () => parseBatchInput(batchInput, batchMode),
        [batchInput, batchMode],
    );

    const namedCount = useMemo(
        () => parsedBatch.items.filter((it) => it.filename !== null).length,
        [parsedBatch.items],
    );

    const downloadBatchZip = useCallback((results: BatchResultItem[]) => {
        const succeeded = results.filter((r) => r.status === "ok" && r.wavBytes && r.filename);
        if (succeeded.length === 0) return;

        const manifest = {
            persona_id: personaId,
            persona_name: personaName,
            created_at: new Date().toISOString(),
            total: results.length,
            succeeded: succeeded.length,
            failed: results.length - succeeded.length,
            items: results.map((r) =>
                r.status === "ok"
                    ? {
                        index: r.index,
                        status: "ok",
                        persona_id: r.personaId,
                        filename: r.filename,
                        text: r.text,
                        applied_text: r.appliedText ?? r.text,
                        engine: r.engine ?? "",
                        duration_sec: r.durationSec ?? null,
                        sample_rate: r.sampleRate ?? null,
                    }
                    : {
                        index: r.index,
                        status: "error",
                        persona_id: r.personaId,
                        filename: null,
                        text: r.text,
                        error: r.error ?? "unknown error",
                    },
            ),
        };
        const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest, null, 2));
        const zipEntries = [
            ...succeeded.map((r) => ({
                name: r.filename!,
                data: r.wavBytes!,
            })),
            { name: "manifest.json", data: manifestBytes },
        ];
        const zipBytes = buildZipBytes(zipEntries);
        // Copy into a plain ArrayBuffer so Blob typing is strict-safe across TS libs
        const ab = new ArrayBuffer(zipBytes.byteLength);
        new Uint8Array(ab).set(zipBytes);
        const zipBlob = new Blob([ab], { type: "application/zip" });
        const safeBase = sanitizeWavFilename(personaName || personaId, "voices").slice(0, -4);
        triggerDownload(zipBlob, `${safeBase}_voices_${formatTimestamp()}.zip`);
    }, [personaId, personaName]);

    const runBatchSynthesis = useCallback(async () => {
        if (parsedBatch.error) {
            setBatchError(parsedBatch.error);
            return;
        }
        if (parsedBatch.items.length === 0) {
            setBatchError("合成するテキストを入力してください。");
            return;
        }

        revokeBatchResults();
        setBatchError(null);
        setBatchCancelled(false);
        setBatchBusy(true);

        const controller = new AbortController();
        batchAbortRef.current = controller;

        const items = parsedBatch.items;
        const padWidth = Math.max(3, String(items.length).length);
        const usedNames = new Set<string>(["manifest.json"]);
        const collected: BatchResultItem[] = [];

        try {
            await flushPendingSaves?.();

            for (let i = 0; i < items.length; i++) {
                if (controller.signal.aborted) {
                    setBatchCancelled(true);
                    break;
                }
                const item = items[i];
                const targetPersona = item.personaId || personaId;
                setBatchProgress({ current: i + 1, total: items.length });

                const fallbackStem = String(item.index).padStart(padWidth, "0");
                const wavName = allocateUniqueWavName(
                    sanitizeWavFilename(item.filename, fallbackStem),
                    usedNames,
                );

                try {
                    const res = await fetch(`${addonApiBase}/synthesize`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        signal: controller.signal,
                        body: JSON.stringify({
                            persona_id: targetPersona,
                            text: item.text,
                            apply_pronunciation: applyPronunciation,
                            clean_text: true,
                            filename: wavName,
                        }),
                    });
                    if (!res.ok) {
                        throw new Error(await extractErrorDetail(res));
                    }
                    const arrayBuf = await res.arrayBuffer();
                    const wavBytes = new Uint8Array(arrayBuf);
                    const blob = new Blob([arrayBuf], { type: "audio/wav" });
                    const objectUrl = URL.createObjectURL(blob);
                    batchUrlsRef.current.push(objectUrl);

                    const engine = res.headers.get("X-TTS-Engine") ?? "";
                    const durRaw = res.headers.get("X-TTS-Duration-Sec");
                    const srRaw = res.headers.get("X-TTS-Sample-Rate");
                    const appliedRaw = res.headers.get("X-TTS-Applied-Text") ?? "";
                    let appliedText = item.text;
                    if (appliedRaw) {
                        try {
                            appliedText = decodeURIComponent(appliedRaw);
                        } catch {
                            appliedText = appliedRaw;
                        }
                    }

                    collected.push({
                        index: item.index,
                        status: "ok",
                        personaId: targetPersona,
                        filename: wavName,
                        text: item.text,
                        appliedText,
                        engine,
                        durationSec: durRaw ? Number(durRaw) : null,
                        sampleRate: srRaw ? Number(srRaw) : null,
                        wavBytes,
                        objectUrl,
                    });
                } catch (err) {
                    if (controller.signal.aborted) {
                        setBatchCancelled(true);
                        break;
                    }
                    collected.push({
                        index: item.index,
                        status: "error",
                        personaId: targetPersona,
                        filename: null,
                        text: item.text,
                        error: err instanceof Error ? err.message : String(err),
                    });
                }
                setBatchResults([...collected]);
            }

            const okCount = collected.filter((r) => r.status === "ok").length;
            if (okCount === 0 && !controller.signal.aborted) {
                setBatchError("すべての項目の合成に失敗しました。各行のエラー内容を確認してください。");
            } else if (okCount > 0 && !controller.signal.aborted) {
                downloadBatchZip(collected);
            }
        } catch (err) {
            if (!controller.signal.aborted) {
                setBatchError(err instanceof Error ? err.message : String(err));
            }
        } finally {
            batchAbortRef.current = null;
            setBatchBusy(false);
            setBatchProgress(null);
        }
    }, [
        addonApiBase,
        applyPronunciation,
        downloadBatchZip,
        flushPendingSaves,
        parsedBatch,
        personaId,
        revokeBatchResults,
    ]);

    const handleBatchFileLoad = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const lower = file.name.toLowerCase();
        if (lower.endsWith(".json")) {
            setBatchMode("json");
        } else if (lower.endsWith(".txt")) {
            setBatchMode("lines");
        }
        const reader = new FileReader();
        reader.onload = () => {
            setBatchInput(typeof reader.result === "string" ? reader.result : "");
            setBatchError(null);
        };
        reader.readAsText(file, "utf-8");
        e.target.value = "";
    };

    const handlePreviewBatchItem = (item: BatchResultItem) => {
        if (!item.objectUrl) return;
        const el = batchAudioRef.current;
        if (!el) return;
        if (previewingIndex === item.index && !el.paused) {
            el.pause();
            setPreviewingIndex(null);
            return;
        }
        el.src = item.objectUrl;
        setPreviewingIndex(item.index);
        void el.play().catch(() => setPreviewingIndex(null));
    };

    const batchSucceededCount = batchResults.filter((r) => r.status === "ok").length;
    const batchFailedCount = batchResults.filter((r) => r.status === "error").length;

    return (
        <div style={styles.root}>
            <div style={styles.headerRow}>
                <div style={styles.titleGroup}>
                    <span style={styles.title}>音声テスト・ファイル出力</span>
                    <span style={styles.personaBadge}>{personaName}</span>
                </div>
                <label style={styles.checkboxLabel}>
                    <input
                        type="checkbox"
                        checked={applyPronunciation}
                        onChange={(e) => setApplyPronunciation(e.target.checked)}
                    />
                    読み方辞書を適用
                </label>
            </div>

            <div style={styles.tabRow}>
                <button
                    type="button"
                    style={{
                        ...styles.tabBtn,
                        ...(tab === "single" ? styles.tabBtnActive : {}),
                    }}
                    onClick={() => setTab("single")}
                >
                    単発テスト・WAV 保存
                </button>
                <button
                    type="button"
                    style={{
                        ...styles.tabBtn,
                        ...(tab === "batch" ? styles.tabBtnActive : {}),
                    }}
                    onClick={() => setTab("batch")}
                >
                    複数テキスト一括 ZIP 出力
                </button>
            </div>

            {tab === "single" && (
                <div style={styles.body}>
                    <textarea
                        style={styles.textarea}
                        rows={3}
                        placeholder="ここにテストしたいセリフを入力 (Ctrl+Enter で生成して再生)"
                        value={singleText}
                        disabled={singleBusy}
                        onChange={(e) => setSingleText(e.target.value)}
                        onKeyDown={(e) => {
                            if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !singleBusy) {
                                e.preventDefault();
                                void runSingleSynthesis({ autoPlay: true, download: false });
                            }
                        }}
                    />

                    <div style={styles.singleControlsRow}>
                        <input
                            type="text"
                            style={styles.filenameInput}
                            placeholder="保存ファイル名 (任意、省略時は自動採番)"
                            value={singleFilename}
                            disabled={singleBusy}
                            onChange={(e) => setSingleFilename(e.target.value)}
                        />
                        <button
                            type="button"
                            style={styles.primaryBtn}
                            disabled={singleBusy || !singleText.trim()}
                            onClick={() => void runSingleSynthesis({ autoPlay: true, download: false })}
                        >
                            {singleBusy ? "合成中…" : "生成して再生"}
                        </button>
                        <button
                            type="button"
                            style={styles.secondaryBtn}
                            disabled={singleBusy || !singleText.trim()}
                            onClick={() => void runSingleSynthesis({ autoPlay: false, download: true })}
                        >
                            WAV を保存
                        </button>
                    </div>

                    {singleError && <div style={styles.errorBox}>{singleError}</div>}

                    {singleResult && (
                        <div style={styles.resultBox}>
                            <audio
                                ref={audioRef}
                                controls
                                src={singleResult.objectUrl}
                                style={styles.audioPlayer}
                            >
                                <track kind="captions" />
                            </audio>
                            <div style={styles.metaRow}>
                                {singleResult.engine && <span>エンジン: {singleResult.engine}</span>}
                                {singleResult.durationSec !== null && (
                                    <span>長さ: {singleResult.durationSec.toFixed(2)} 秒</span>
                                )}
                                {singleResult.sampleRate !== null && (
                                    <span>{singleResult.sampleRate} Hz / WAV</span>
                                )}
                            </div>
                            {singleResult.appliedText !== singleResult.text && (
                                <div style={styles.appliedTextHint}>
                                    読み上げ文 (辞書/整形後): {singleResult.appliedText}
                                </div>
                            )}
                            {isSingleResultStale && (
                                <div style={styles.appliedTextHint}>
                                    ※設定またはテキストが変更されています (次の再生・WAV 保存で新しく合成されます)
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {tab === "batch" && (
                <div style={styles.body}>
                    <div style={styles.batchToolbar}>
                        <div style={styles.modeToggleGroup}>
                            <button
                                type="button"
                                style={{
                                    ...styles.modeBtn,
                                    ...(batchMode === "json" ? styles.modeBtnActive : {}),
                                }}
                                disabled={batchBusy}
                                onClick={() => setBatchMode("json")}
                            >
                                JSON 配列
                            </button>
                            <button
                                type="button"
                                style={{
                                    ...styles.modeBtn,
                                    ...(batchMode === "lines" ? styles.modeBtnActive : {}),
                                }}
                                disabled={batchBusy}
                                onClick={() => setBatchMode("lines")}
                            >
                                1 行 1 セリフ
                            </button>
                        </div>

                        <div style={styles.toolbarRight}>
                            <button
                                type="button"
                                style={styles.smallBtn}
                                disabled={batchBusy}
                                onClick={() => batchFileInputRef.current?.click()}
                            >
                                ファイル読込 (.json / .txt)
                            </button>
                            <button
                                type="button"
                                style={styles.smallBtn}
                                disabled={batchBusy}
                                onClick={() => {
                                    setBatchInput(batchMode === "json" ? SAMPLE_JSON : SAMPLE_LINES);
                                    setBatchError(null);
                                }}
                            >
                                例を入れる
                            </button>
                            <input
                                ref={batchFileInputRef}
                                type="file"
                                accept=".json,.txt,application/json,text/plain"
                                style={{ display: "none" }}
                                onChange={handleBatchFileLoad}
                            />
                        </div>
                    </div>

                    <div style={styles.hintText}>
                        {batchMode === "json"
                            ? '文字列の配列 ["文1", "文2"]、または [{"filename": "名前", "text": "セリフ"}] 形式。要素ごとに "persona_id" で別ペルソナを指定することもできます。ZIP 内に manifest.json を同梱します。'
                            : "改行で区切って 1 行につき 1 セリフを入力 (空行は無視)。英語文のカンマでも誤爆しません。ファイル名は 001.wav から自動連番になります。"}
                    </div>

                    <textarea
                        style={styles.codeTextarea}
                        rows={7}
                        placeholder={batchMode === "json" ? SAMPLE_JSON : SAMPLE_LINES}
                        value={batchInput}
                        disabled={batchBusy}
                        onChange={(e) => {
                            setBatchInput(e.target.value);
                            setBatchError(null);
                        }}
                    />

                    <div style={styles.parseStatusRow}>
                        {parsedBatch.error ? (
                            <span style={styles.parseErrorText}>{parsedBatch.error}</span>
                        ) : parsedBatch.items.length > 0 ? (
                            <span style={styles.parseOkText}>
                                {parsedBatch.items.length} 件のセリフを認識
                                {namedCount > 0 ? ` (ファイル名指定: ${namedCount} 件)` : ""}
                            </span>
                        ) : (
                            <span style={styles.parseIdleText}>テキストを入力するかファイルを読み込んでください</span>
                        )}
                    </div>

                    <div style={styles.batchActionRow}>
                        {!batchBusy ? (
                            <button
                                type="button"
                                style={styles.primaryBtn}
                                disabled={!!parsedBatch.error || parsedBatch.items.length === 0}
                                onClick={() => void runBatchSynthesis()}
                            >
                                一括合成して ZIP 保存 ({parsedBatch.items.length} 件)
                            </button>
                        ) : (
                            <button
                                type="button"
                                style={styles.dangerBtn}
                                onClick={() => batchAbortRef.current?.abort()}
                            >
                                中止する
                            </button>
                        )}
                        {!batchBusy && batchSucceededCount > 0 && (
                            <button
                                type="button"
                                style={styles.secondaryBtn}
                                onClick={() => downloadBatchZip(batchResults)}
                            >
                                ZIP を再保存 ({batchSucceededCount} 件)
                            </button>
                        )}
                    </div>

                    {batchProgress && (
                        <div style={styles.progressWrap}>
                            <div style={styles.progressLabel}>
                                合成中: {batchProgress.current} / {batchProgress.total} 件
                            </div>
                            <div style={styles.progressTrack}>
                                <div
                                    style={{
                                        ...styles.progressFill,
                                        width: `${Math.round((batchProgress.current / batchProgress.total) * 100)}%`,
                                    }}
                                />
                            </div>
                        </div>
                    )}

                    {batchCancelled && !batchBusy && (
                        <div style={styles.hintText}>
                            途中で中止しました (完了済みの {batchSucceededCount} 件は「ZIP を再保存」から保存できます)。
                        </div>
                    )}

                    {batchError && <div style={styles.errorBox}>{batchError}</div>}

                    {batchResults.length > 0 && (
                        <div style={styles.batchResultsBox}>
                            <div style={styles.batchSummaryRow}>
                                <span>
                                    結果: 成功 {batchSucceededCount} 件
                                    {batchFailedCount > 0 ? ` / 失敗 ${batchFailedCount} 件` : ""}
                                </span>
                            </div>
                            <audio
                                ref={batchAudioRef}
                                style={{ display: "none" }}
                                onEnded={() => setPreviewingIndex(null)}
                            >
                                <track kind="captions" />
                            </audio>
                            <div style={styles.batchList}>
                                {batchResults.map((r) => (
                                    <div key={r.index} style={styles.batchItemRow}>
                                        <span
                                            style={{
                                                ...styles.statusBadge,
                                                ...(r.status === "ok"
                                                    ? styles.statusOk
                                                    : styles.statusErr),
                                            }}
                                        >
                                            {r.status === "ok" ? "OK" : "NG"}
                                        </span>
                                        <span style={styles.batchItemName}>
                                            {r.filename ?? `#${r.index}`}
                                        </span>
                                        <span style={styles.batchItemText} title={r.text}>
                                            {r.status === "ok" ? r.text : r.error}
                                        </span>
                                        {r.status === "ok" && r.durationSec != null && (
                                            <span style={styles.batchItemMeta}>
                                                {r.durationSec.toFixed(1)}s
                                            </span>
                                        )}
                                        {r.status === "ok" && r.objectUrl && (
                                            <button
                                                type="button"
                                                style={styles.previewBtn}
                                                onClick={() => handlePreviewBatchItem(r)}
                                            >
                                                {previewingIndex === r.index ? "停止" : "試聴"}
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

const styles: Record<string, React.CSSProperties> = {
    root: {
        marginTop: "0.75rem",
        padding: "0.85rem",
        borderRadius: "0.6rem",
        border: "1px solid var(--voice-tts-border)",
        background: "var(--voice-tts-bg)",
        color: "var(--voice-tts-text)",
        display: "flex",
        flexDirection: "column",
        gap: "0.6rem",
    },
    headerRow: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "0.5rem",
    },
    titleGroup: {
        display: "flex",
        alignItems: "center",
        gap: "0.45rem",
    },
    title: {
        fontSize: "0.86rem",
        fontWeight: 600,
    },
    personaBadge: {
        fontSize: "0.74rem",
        padding: "0.1rem 0.45rem",
        borderRadius: "999px",
        background: "var(--voice-tts-accent-soft-bg)",
        color: "var(--voice-tts-accent-soft-fg)",
        fontWeight: 500,
    },
    checkboxLabel: {
        display: "inline-flex",
        alignItems: "center",
        gap: "0.35rem",
        fontSize: "0.78rem",
        color: "var(--voice-tts-text-muted)",
        cursor: "pointer",
        userSelect: "none",
    },
    tabRow: {
        display: "flex",
        gap: "0.35rem",
        borderBottom: "1px solid var(--voice-tts-border-subtle)",
        paddingBottom: "0.4rem",
    },
    tabBtn: {
        padding: "0.32rem 0.7rem",
        fontSize: "0.8rem",
        borderRadius: "0.4rem",
        border: "1px solid transparent",
        background: "transparent",
        color: "var(--voice-tts-text-muted)",
        cursor: "pointer",
    },
    tabBtnActive: {
        background: "var(--voice-tts-accent-soft-bg)",
        color: "var(--voice-tts-accent-soft-fg)",
        fontWeight: 600,
    },
    body: {
        display: "flex",
        flexDirection: "column",
        gap: "0.55rem",
    },
    textarea: {
        width: "100%",
        boxSizing: "border-box",
        padding: "0.5rem 0.6rem",
        borderRadius: "0.45rem",
        border: "1px solid var(--voice-tts-border)",
        background: "var(--voice-tts-surface)",
        color: "var(--voice-tts-text)",
        fontSize: "0.85rem",
        lineHeight: 1.45,
        resize: "vertical",
        fontFamily: "inherit",
    },
    codeTextarea: {
        width: "100%",
        boxSizing: "border-box",
        padding: "0.5rem 0.6rem",
        borderRadius: "0.45rem",
        border: "1px solid var(--voice-tts-border)",
        background: "var(--voice-tts-surface)",
        color: "var(--voice-tts-text)",
        fontSize: "0.8rem",
        lineHeight: 1.45,
        resize: "vertical",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    },
    singleControlsRow: {
        display: "flex",
        flexWrap: "wrap",
        gap: "0.45rem",
        alignItems: "center",
    },
    filenameInput: {
        flex: "1 1 180px",
        minWidth: "140px",
        padding: "0.4rem 0.55rem",
        borderRadius: "0.4rem",
        border: "1px solid var(--voice-tts-border)",
        background: "var(--voice-tts-surface)",
        color: "var(--voice-tts-text)",
        fontSize: "0.8rem",
    },
    primaryBtn: {
        padding: "0.42rem 0.85rem",
        borderRadius: "0.4rem",
        border: "none",
        background: "var(--voice-tts-accent)",
        color: "var(--voice-tts-accent-fg)",
        fontSize: "0.8rem",
        fontWeight: 600,
        cursor: "pointer",
    },
    secondaryBtn: {
        padding: "0.42rem 0.8rem",
        borderRadius: "0.4rem",
        border: "1px solid var(--voice-tts-secondary-border)",
        background: "var(--voice-tts-secondary-bg)",
        color: "var(--voice-tts-secondary-fg)",
        fontSize: "0.8rem",
        fontWeight: 500,
        cursor: "pointer",
    },
    dangerBtn: {
        padding: "0.42rem 0.85rem",
        borderRadius: "0.4rem",
        border: "1px solid var(--voice-tts-err-border)",
        background: "var(--voice-tts-err-bg)",
        color: "var(--voice-tts-err-fg)",
        fontSize: "0.8rem",
        fontWeight: 600,
        cursor: "pointer",
    },
    smallBtn: {
        padding: "0.25rem 0.55rem",
        borderRadius: "0.35rem",
        border: "1px solid var(--voice-tts-secondary-border)",
        background: "var(--voice-tts-secondary-bg)",
        color: "var(--voice-tts-secondary-fg)",
        fontSize: "0.75rem",
        cursor: "pointer",
    },
    errorBox: {
        padding: "0.5rem 0.65rem",
        borderRadius: "0.4rem",
        border: "1px solid var(--voice-tts-err-border)",
        background: "var(--voice-tts-err-bg)",
        color: "var(--voice-tts-err-fg)",
        fontSize: "0.78rem",
        lineHeight: 1.45,
        wordBreak: "break-word",
        overflowWrap: "anywhere",
    },
    resultBox: {
        padding: "0.55rem 0.65rem",
        borderRadius: "0.45rem",
        border: "1px solid var(--voice-tts-border-subtle)",
        background: "var(--voice-tts-surface)",
        display: "flex",
        flexDirection: "column",
        gap: "0.35rem",
    },
    audioPlayer: {
        width: "100%",
        height: "34px",
    },
    metaRow: {
        display: "flex",
        flexWrap: "wrap",
        gap: "0.75rem",
        fontSize: "0.74rem",
        color: "var(--voice-tts-text-subtle)",
    },
    appliedTextHint: {
        fontSize: "0.75rem",
        color: "var(--voice-tts-text-muted)",
        wordBreak: "break-word",
        overflowWrap: "anywhere",
    },
    batchToolbar: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "0.45rem",
    },
    modeToggleGroup: {
        display: "inline-flex",
        borderRadius: "0.4rem",
        border: "1px solid var(--voice-tts-border)",
        overflow: "hidden",
    },
    modeBtn: {
        padding: "0.26rem 0.6rem",
        fontSize: "0.76rem",
        border: "none",
        background: "var(--voice-tts-surface)",
        color: "var(--voice-tts-text-muted)",
        cursor: "pointer",
    },
    modeBtnActive: {
        background: "var(--voice-tts-accent)",
        color: "var(--voice-tts-accent-fg)",
        fontWeight: 600,
    },
    toolbarRight: {
        display: "flex",
        gap: "0.35rem",
        flexWrap: "wrap",
    },
    hintText: {
        fontSize: "0.75rem",
        color: "var(--voice-tts-text-subtle)",
        lineHeight: 1.45,
    },
    parseStatusRow: {
        fontSize: "0.76rem",
    },
    parseOkText: {
        color: "var(--voice-tts-accent-soft-fg)",
        fontWeight: 500,
    },
    parseErrorText: {
        color: "var(--voice-tts-err-fg)",
        fontWeight: 500,
    },
    parseIdleText: {
        color: "var(--voice-tts-text-subtle)",
    },
    batchActionRow: {
        display: "flex",
        flexWrap: "wrap",
        gap: "0.45rem",
        alignItems: "center",
    },
    progressWrap: {
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
    },
    progressLabel: {
        fontSize: "0.76rem",
        color: "var(--voice-tts-text-muted)",
    },
    progressTrack: {
        width: "100%",
        height: "6px",
        borderRadius: "999px",
        background: "var(--voice-tts-progress-track)",
        overflow: "hidden",
    },
    progressFill: {
        height: "100%",
        background: "var(--voice-tts-progress-fill)",
        transition: "width 0.15s ease",
    },
    batchResultsBox: {
        borderRadius: "0.45rem",
        border: "1px solid var(--voice-tts-border-subtle)",
        background: "var(--voice-tts-surface)",
        padding: "0.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.35rem",
    },
    batchSummaryRow: {
        fontSize: "0.78rem",
        fontWeight: 600,
        color: "var(--voice-tts-text-muted)",
    },
    batchList: {
        maxHeight: "200px",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
    },
    batchItemRow: {
        display: "flex",
        alignItems: "center",
        gap: "0.45rem",
        fontSize: "0.76rem",
        padding: "0.2rem 0.25rem",
        borderBottom: "1px solid var(--voice-tts-border-subtle)",
    },
    statusBadge: {
        padding: "0.08rem 0.35rem",
        borderRadius: "0.25rem",
        fontSize: "0.7rem",
        fontWeight: 700,
        flexShrink: 0,
    },
    statusOk: {
        background: "var(--voice-tts-ok-bg)",
        color: "var(--voice-tts-ok-fg)",
    },
    statusErr: {
        background: "var(--voice-tts-err-bg)",
        color: "var(--voice-tts-err-fg)",
    },
    batchItemName: {
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: "0.74rem",
        flexShrink: 0,
        color: "var(--voice-tts-text)",
    },
    batchItemText: {
        flex: 1,
        minWidth: 0,
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        color: "var(--voice-tts-text-muted)",
    },
    batchItemMeta: {
        fontSize: "0.72rem",
        color: "var(--voice-tts-text-subtle)",
        flexShrink: 0,
    },
    previewBtn: {
        padding: "0.15rem 0.45rem",
        borderRadius: "0.3rem",
        border: "1px solid var(--voice-tts-secondary-border)",
        background: "var(--voice-tts-secondary-bg)",
        color: "var(--voice-tts-secondary-fg)",
        fontSize: "0.72rem",
        cursor: "pointer",
        flexShrink: 0,
    },
};
