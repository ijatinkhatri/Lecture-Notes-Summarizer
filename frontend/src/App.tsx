import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { UploadZone } from './components/UploadZone';
import { ResultsView, ExportFormat } from './components/ResultsView';
import { SettingsModal } from './components/SettingsModal';
import { api } from './lib/api';
import { SAMPLE_LECTURE_TITLE, SAMPLE_LECTURE_TEXT } from './lib/sampleLecture';
import {
  DocumentPreview as IDocumentPreview,
  StudyPackageOutput,
  SettingsDTO,
} from './types';
import { Loader2, AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  // Theme state
  const [isDark, setIsDark] = useState<boolean>(() => {
    return (
      localStorage.getItem('notenest_theme') === 'dark' ||
      (!('notenest_theme' in localStorage) &&
        window.matchMedia('(prefers-color-scheme: dark)').matches)
    );
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('notenest_theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('notenest_theme', 'light');
    }
  }, [isDark]);

  // Core State
  const [currentDocument, setCurrentDocument] = useState<IDocumentPreview | null>(null);
  const [summaryOutput, setSummaryOutput] = useState<StudyPackageOutput | null>(null);
  const [summaryMode, setSummaryMode] = useState<'short' | 'detailed'>('detailed');
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [settings, setSettings] = useState<SettingsDTO>({
    has_gemini_api_key: false,
    gemini_api_key_masked: '',
    active_model: 'gemini-3.5-flash-lite',
    temperature: 0.2,
    default_summary_mode: 'detailed',
    default_audience: 'intermediate',
    offline_available: true,
    available_models: [],
  });

  useEffect(() => {
    api
      .getSettings()
      .then((s) => setSettings(s))
      .catch((e) => console.error('Failed to load settings', e));
  }, []);

  // Summarize a given document immediately
  const runSummarize = async (doc: IDocumentPreview, mode: 'short' | 'detailed') => {
    setErrorMessage(null);
    setIsProcessing(true);
    try {
      const response = await api.summarize({
        document_id: doc.id,
        summary_mode: mode,
        audience_level: 'intermediate',
        language: 'English',
        generate_summary: true,
        generate_concepts: true,
        generate_revision_points: true,
        generate_flashcards: false,
      });
      setSummaryOutput(response.output);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to generate summary. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Upload & Auto-Summarize
  const handleUploadFile = async (file: File): Promise<IDocumentPreview> => {
    setErrorMessage(null);
    const doc = await api.uploadFile(file);
    return doc;
  };

  const handlePasteText = async (title: string, text: string): Promise<IDocumentPreview> => {
    setErrorMessage(null);
    const doc = await api.pasteText(title, text);
    return doc;
  };

  const handleDocumentLoaded = async (doc: IDocumentPreview) => {
    setCurrentDocument(doc);
    setSummaryOutput(null);
    await runSummarize(doc, summaryMode);
  };

  const handleLoadSample = async () => {
    setErrorMessage(null);
    setIsProcessing(true);
    try {
      const doc = await api.pasteText(SAMPLE_LECTURE_TITLE, SAMPLE_LECTURE_TEXT);
      setCurrentDocument(doc);
      setSummaryOutput(null);
      await runSummarize(doc, summaryMode);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to load sample lecture.');
      setIsProcessing(false);
    }
  };

  const handleChangeMode = async (newMode: 'short' | 'detailed') => {
    setSummaryMode(newMode);
    if (currentDocument) {
      await runSummarize(currentDocument, newMode);
    }
  };

  const handleExport = async (format: ExportFormat) => {
    if (!summaryOutput || !currentDocument) return;
    try {
      setErrorMessage(null);
      await api.exportFile(format, currentDocument.title, 'study_package', summaryOutput);
    } catch (err: any) {
      setErrorMessage(err.message || `Failed to download ${format.toUpperCase()} file.`);
    }
  };

  const handleReset = () => {
    setCurrentDocument(null);
    setSummaryOutput(null);
    setErrorMessage(null);
  };

  const handleUpdateSettings = async (payload: any) => {
    const updated = await api.updateSettings(payload);
    setSettings((prev) => ({
      ...prev,
      has_gemini_api_key: updated.has_gemini_api_key,
      gemini_api_key_masked: updated.gemini_api_key_masked,
      active_model: updated.active_model,
      temperature: updated.temperature,
    }));
  };

  return (
    <div className="min-h-screen flex flex-col bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100 transition-colors">
      <Header
        onReset={handleReset}
        onOpenSettings={() => setSettingsOpen(true)}
        isDark={isDark}
        setIsDark={setIsDark}
        hasApiKey={settings.has_gemini_api_key}
      />

      <main className="flex-1 mx-auto w-full max-w-4xl px-4 sm:px-6 py-8 sm:py-10">
        {/* Error Banner */}
        {errorMessage && (
          <div className="mb-6 flex items-center justify-between gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-300">
            <div className="flex items-center gap-2 font-medium">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 dark:text-rose-400" />
              <span>{errorMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              className="text-xs font-semibold text-rose-600 hover:text-rose-800 dark:text-rose-400"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* State 1: Upload View */}
        {!currentDocument && (
          <div>
            <Hero />
            <UploadZone
              onDocumentLoaded={handleDocumentLoaded}
              onUploadFile={handleUploadFile}
              onPasteText={handlePasteText}
              onLoadSample={handleLoadSample}
              summaryMode={summaryMode}
              setSummaryMode={setSummaryMode}
              isLoading={isProcessing}
            />
          </div>
        )}

        {/* State 2: Initial Summarizing Loader */}
        {currentDocument && !summaryOutput && isProcessing && (
          <div className="rounded-2xl border border-zinc-200/90 bg-white p-12 text-center shadow-xs dark:border-zinc-800 dark:bg-zinc-900/60">
            <Loader2 className="mx-auto h-9 w-9 animate-spin text-zinc-900 dark:text-zinc-100" />
            <h2 className="mt-4 text-lg font-semibold text-zinc-950 dark:text-zinc-50">
              Summarizing "{currentDocument.title}"
            </h2>
            <p className="mt-1.5 text-sm text-zinc-500 dark:text-zinc-400">
              Extracting key points from {currentDocument.page_count}{' '}
              {currentDocument.file_type.toLowerCase().includes('ppt') ? 'slides' : 'pages'}...
            </p>
          </div>
        )}

        {/* State 3: Summarized Output Ready to Read & Download */}
        {currentDocument && summaryOutput && (
          <ResultsView
            document={currentDocument}
            output={summaryOutput}
            summaryMode={summaryMode}
            onChangeMode={handleChangeMode}
            onExport={handleExport}
            onNewUpload={handleReset}
            isRegenerating={isProcessing}
          />
        )}
      </main>

      {/* Settings Modal (Optional API Key) */}
      {settingsOpen && (
        <SettingsModal
          settings={settings}
          onUpdateSettings={handleUpdateSettings}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      {/* Minimal Footer */}
      <footer className="border-t border-zinc-200/80 bg-white/70 py-5 dark:border-zinc-800/80 dark:bg-zinc-950/70 mt-auto">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
          <span>
            <strong className="font-semibold text-zinc-800 dark:text-zinc-200">NoteNest</strong> — Upload PPT or PDF, read summary, download in any format.
          </span>
          <button
            type="button"
            onClick={() => setSettingsOpen(true)}
            className="hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors"
          >
            API Settings
          </button>
        </div>
      </footer>
    </div>
  );
};

export default App;
