import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { Upload, Camera, Loader2, CheckCircle, XCircle } from "lucide-react";

export default function App() {
  const [loading, setLoading] = useState(true); // Startup state
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [status, setStatus] = useState("idle"); // idle, processing, success, error

  // Simulate Startup Animation
  useEffect(() => {
    setTimeout(() => setLoading(false), 2000);
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setStatus("idle");
      setCaption("");
    }
  };

  const generateCaption = async () => {
    if (!image) return;
    setStatus("processing");
    
    const formData = new FormData();
    formData.append("file", image);

    try {
      // Connect to FastAPI backend
      const res = await axios.post("http://localhost:8000/predict", formData);
      if (res.data.status === "success") {
        setCaption(res.data.caption);
        setStatus("success");
      } else {
        setStatus("error");
      }
    } catch (err) {
      setStatus("error");
    }
  };

  // --- 1. STARTUP SCREEN ---
  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-900">
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, type: "spring" }}
          className="flex flex-col items-center"
        >
          <Camera size={64} className="text-pink-500 mb-4" />
          <h1 className="text-3xl font-bold text-white tracking-widest">INSTACAP AI</h1>
        </motion.div>
      </div>
    );
  }

  // --- 2. MAIN APP UI ---
  return (
    <div className="min-h-screen bg-slate-900 p-6 flex flex-col items-center justify-center">
      <motion.div 
        initial={{ y: 20, opacity: 0 }} 
        animate={{ y: 0, opacity: 1 }}
        className="w-full max-w-md bg-slate-800 rounded-2xl shadow-2xl p-6 border border-slate-700"
      >
        <div className="flex items-center justify-center mb-6">
          <h2 className="text-xl font-bold bg-gradient-to-r from-pink-500 to-violet-500 bg-clip-text text-transparent">
            Instagram Caption Generator
          </h2>
        </div>

        {/* IMAGE UPLOAD AREA */}
        <div className="relative group">
          <input 
            type="file" 
            onChange={handleFileChange} 
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          />
          <div className={`
            border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all
            ${preview ? 'border-violet-500 bg-slate-900' : 'border-slate-600 hover:border-slate-400 bg-slate-800'}
          `}>
            {preview ? (
              <motion.img 
                layoutId="preview"
                src={preview} 
                alt="Preview" 
                className="rounded-lg shadow-lg max-h-64 object-cover"
              />
            ) : (
              <>
                <Upload className="text-slate-400 mb-2" size={40} />
                <p className="text-slate-400 text-sm">Tap to upload image</p>
              </>
            )}
          </div>
        </div>

        {/* ACTION BUTTON */}
        <div className="mt-6">
          {status === "processing" ? (
            <button disabled className="w-full bg-slate-700 text-slate-300 font-bold py-3 px-4 rounded-xl flex items-center justify-center gap-2 cursor-not-allowed">
              <Loader2 className="animate-spin" /> Generating...
            </button>
          ) : (
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={generateCaption}
              disabled={!image}
              className={`
                w-full font-bold py-3 px-4 rounded-xl shadow-lg transition-colors flex items-center justify-center gap-2
                ${!image ? 'bg-slate-700 text-slate-500 cursor-not-allowed' : 'bg-gradient-to-r from-pink-600 to-violet-600 text-white hover:from-pink-500 hover:to-violet-500'}
              `}
            >
               ✨ Generate Caption
            </motion.button>
          )}
        </div>

        {/* RESULT AREA */}
        <AnimatePresence>
          {status === "success" && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6 bg-slate-900 rounded-xl p-4 border border-green-500/30"
            >
              <div className="flex items-center gap-2 mb-2 text-green-400">
                <CheckCircle size={18} />
                <span className="text-sm font-bold">Caption Ready!</span>
              </div>
              <p className="text-lg text-slate-200 font-medium italic">"{caption}"</p>
              <button 
                onClick={() => navigator.clipboard.writeText(caption)}
                className="mt-3 text-xs text-slate-500 hover:text-white transition-colors"
              >
                Copy to clipboard
              </button>
            </motion.div>
          )}

          {status === "error" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mt-6 bg-red-900/20 rounded-xl p-4 border border-red-500/30 flex items-center gap-3 text-red-400"
            >
              <XCircle size={24} />
              <div>
                <p className="font-bold text-sm">Generation Failed</p>
                <p className="text-xs opacity-80">Try a different image.</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}