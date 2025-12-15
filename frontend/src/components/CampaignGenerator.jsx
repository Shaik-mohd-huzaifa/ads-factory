import { useState, useRef } from 'react'
import { Sparkles, FileText, Upload, Plus, Wand2, Layers, Zap, X, Image, FileImage, CheckCircle } from 'lucide-react'

const API_BASE = 'http://localhost:8001'

export function CampaignGenerator({
    prompt,
    setPrompt,
    selectedBrand,
    handleGenerate,
    isGenerating,
    platforms,
    selectedPlatforms,
    togglePlatform,
    setShowBrandModal,
    activeTab,
    setActiveTab,
    onAssetsChange
}) {
    const [uploadedAssets, setUploadedAssets] = useState([])
    const [isUploading, setIsUploading] = useState(false)
    const [uploadError, setUploadError] = useState(null)
    const fileInputRef = useRef(null)

    const handleFileSelect = async (e) => {
        const files = Array.from(e.target.files)
        if (!files.length || !selectedBrand) return

        setIsUploading(true)
        setUploadError(null)

        for (const file of files) {
            try {
                const formData = new FormData()
                formData.append('file', file)

                const response = await fetch(`${API_BASE}/api/brands/${selectedBrand.id}/assets`, {
                    method: 'POST',
                    body: formData,
                })

                if (response.ok) {
                    const asset = await response.json()
                    setUploadedAssets(prev => {
                        const newAssets = [...prev, {
                            ...asset,
                            preview: URL.createObjectURL(file)
                        }]
                        // Notify parent of asset IDs change
                        if (onAssetsChange) {
                            onAssetsChange(newAssets.map(a => a.id))
                        }
                        return newAssets
                    })
                } else {
                    const error = await response.text()
                    setUploadError(`Failed to upload ${file.name}: ${error}`)
                }
            } catch (err) {
                setUploadError(`Error uploading ${file.name}: ${err.message}`)
            }
        }

        setIsUploading(false)
        if (fileInputRef.current) fileInputRef.current.value = ''
    }

    const removeAsset = (assetId) => {
        setUploadedAssets(prev => {
            const newAssets = prev.filter(a => a.id !== assetId)
            // Notify parent of asset IDs change
            if (onAssetsChange) {
                onAssetsChange(newAssets.map(a => a.id))
            }
            return newAssets
        })
    }
    return (
        <div className="max-w-4xl mx-auto mb-8">
            {/* Hero Section */}
            <div className="text-center mb-8">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium mb-4">
                    <Zap className="w-4 h-4" />
                    Powered by Gemini 2.0 Flash + Vector Search
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Create Multi-Platform Campaigns</h2>
                <p className="text-gray-500 max-w-lg mx-auto">
                    Describe your campaign idea and we'll generate consistent designs across all platforms using AI
                </p>
            </div>

            <div className="bg-white rounded-2xl border border-gray-200 shadow-lg overflow-hidden">
                {/* Tabs */}
                <div className="flex border-b border-gray-200 bg-gray-50">
                    <button
                        onClick={() => setActiveTab('generate')}
                        className={`flex items-center gap-2 px-6 py-4 text-sm font-medium transition-all ${activeTab === 'generate'
                                ? 'bg-white border-b-2 border-indigo-600 text-indigo-600 -mb-px'
                                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <Wand2 className="w-4 h-4" />
                        AI Generate
                    </button>
                    <button
                        onClick={() => setActiveTab('templates')}
                        className={`flex items-center gap-2 px-6 py-4 text-sm font-medium transition-all ${activeTab === 'templates'
                                ? 'bg-white border-b-2 border-indigo-600 text-indigo-600 -mb-px'
                                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <Layers className="w-4 h-4" />
                        Templates
                    </button>
                    <button
                        onClick={() => setActiveTab('upload')}
                        className={`flex items-center gap-2 px-6 py-4 text-sm font-medium transition-all ${activeTab === 'upload'
                                ? 'bg-white border-b-2 border-indigo-600 text-indigo-600 -mb-px'
                                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <Upload className="w-4 h-4" />
                        Upload Assets
                    </button>
                </div>

                <div className="p-6">
                    {/* Brand Selector */}
                    {selectedBrand ? (
                        <div className="flex items-center gap-3 mb-5 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border border-gray-200">
                            <div
                                className="w-10 h-10 rounded-xl shadow-sm flex items-center justify-center text-white font-bold text-lg"
                                style={{ backgroundColor: selectedBrand.primary_color }}
                            >
                                {selectedBrand.name?.charAt(0).toUpperCase()}
                            </div>
                            <div className="flex-1">
                                <p className="font-semibold text-gray-900">{selectedBrand.name}</p>
                                <p className="text-sm text-gray-500">{selectedBrand.industry} • {selectedBrand.tone || 'Professional'}</p>
                            </div>
                            <button 
                                onClick={() => setShowBrandModal(true)}
                                className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
                            >
                                Change
                            </button>
                        </div>
                    ) : (
                        <button
                            onClick={() => setShowBrandModal(true)}
                            className="w-full mb-5 p-4 border-2 border-dashed border-gray-300 rounded-xl text-gray-500 hover:border-indigo-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all flex items-center justify-center gap-2"
                        >
                            <Plus className="w-5 h-5" />
                            Create a brand kit first
                        </button>
                    )}

                    {/* Prompt Input */}
                    <div className="relative">
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Describe your campaign idea...

Example: 'Black Friday Sale - 50% off all AI products with futuristic tech vibes, dark theme with neon accents'"
                            className="w-full h-36 p-5 bg-gray-50 rounded-xl border border-gray-200 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-gray-700 placeholder-gray-400 text-base"
                        />
                        <div className="absolute bottom-3 right-3 text-xs text-gray-400">
                            {prompt.length} characters
                        </div>
                    </div>

                    {/* Asset Upload Section */}
                    <div className="mt-5 p-4 bg-gray-50 rounded-xl border border-gray-200">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                                <FileImage className="w-4 h-4 text-gray-600" />
                                <p className="text-sm font-semibold text-gray-700">Reference Assets</p>
                                <span className="text-xs text-gray-400">(optional)</span>
                            </div>
                            {uploadedAssets.length > 0 && (
                                <span className="text-xs text-green-600 flex items-center gap-1">
                                    <CheckCircle className="w-3 h-3" />
                                    {uploadedAssets.length} uploaded
                                </span>
                            )}
                        </div>
                        
                        {/* Upload Area */}
                        <div className="flex flex-wrap gap-3">
                            {/* Uploaded Assets Preview */}
                            {uploadedAssets.map((asset) => (
                                <div key={asset.id} className="relative group">
                                    <div className="w-20 h-20 rounded-lg overflow-hidden border-2 border-gray-200 bg-white">
                                        {asset.preview ? (
                                            <img src={asset.preview} alt={asset.filename} className="w-full h-full object-cover" />
                                        ) : (
                                            <div className="w-full h-full flex items-center justify-center bg-gray-100">
                                                <FileText className="w-6 h-6 text-gray-400" />
                                            </div>
                                        )}
                                    </div>
                                    <button
                                        onClick={() => removeAsset(asset.id)}
                                        className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-md"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                    <p className="text-xs text-gray-500 mt-1 truncate w-20">{asset.filename}</p>
                                </div>
                            ))}
                            
                            {/* Upload Button */}
                            <label className={`w-20 h-20 rounded-lg border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-all ${
                                !selectedBrand 
                                    ? 'border-gray-200 bg-gray-50 cursor-not-allowed' 
                                    : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50'
                            }`}>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    multiple
                                    accept="image/*"
                                    onChange={handleFileSelect}
                                    disabled={!selectedBrand || isUploading}
                                    className="hidden"
                                />
                                {isUploading ? (
                                    <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                                ) : (
                                    <>
                                        <Upload className={`w-5 h-5 ${selectedBrand ? 'text-gray-400' : 'text-gray-300'}`} />
                                        <span className={`text-xs mt-1 ${selectedBrand ? 'text-gray-500' : 'text-gray-300'}`}>Add</span>
                                    </>
                                )}
                            </label>
                        </div>
                        
                        {uploadError && (
                            <p className="text-xs text-red-500 mt-2">{uploadError}</p>
                        )}
                        
                        <p className="text-xs text-gray-400 mt-2">
                            Upload logos, product images, or reference designs. These will be stored and used for AI matching.
                        </p>
                    </div>

                    {/* Platform Selector */}
                    <div className="mt-5">
                        <div className="flex items-center justify-between mb-3">
                            <p className="text-sm font-semibold text-gray-700">Target Platforms</p>
                            <button 
                                onClick={() => {
                                    if (selectedPlatforms.length === platforms.length) {
                                        // Deselect all
                                    } else {
                                        // Select all would need prop function
                                    }
                                }}
                                className="text-xs text-indigo-600 hover:text-indigo-700 font-medium"
                            >
                                {selectedPlatforms.length} of {platforms.length} selected
                            </button>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            {platforms.map((platform) => {
                                const Icon = platform.icon
                                const isSelected = selectedPlatforms.includes(platform.id)
                                return (
                                    <button
                                        key={platform.id}
                                        onClick={() => togglePlatform(platform.id)}
                                        className={`flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-medium transition-all border-2 ${isSelected
                                                ? 'bg-indigo-50 text-indigo-700 border-indigo-500 shadow-sm'
                                                : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        <Icon className="w-4 h-4" />
                                        <span className="flex-1 text-left">{platform.name}</span>
                                        <span className={`text-xs px-1.5 py-0.5 rounded ${isSelected ? 'bg-indigo-200 text-indigo-700' : 'bg-gray-100 text-gray-500'}`}>
                                            {platform.ratio}
                                        </span>
                                    </button>
                                )
                            })}
                        </div>
                    </div>

                    {/* Generate Button */}
                    <div className="flex items-center justify-between mt-6 pt-6 border-t border-gray-100">
                        <div className="text-sm text-gray-500">
                            <span className="font-medium text-gray-700">{selectedPlatforms.length}</span> platforms selected
                        </div>
                        <button
                            onClick={handleGenerate}
                            disabled={!prompt.trim() || !selectedBrand || isGenerating}
                            className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl disabled:shadow-none"
                        >
                            {isGenerating ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    <span>Generating Campaign...</span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-5 h-5" />
                                    <span>Generate Campaign</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Features */}
            <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-200">
                    <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                        <Sparkles className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                        <p className="font-medium text-gray-900 text-sm">AI-Powered</p>
                        <p className="text-xs text-gray-500">Gemini 2.0 Flash</p>
                    </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-200">
                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                        <Layers className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                        <p className="font-medium text-gray-900 text-sm">Vector Search</p>
                        <p className="text-xs text-gray-500">Smart asset matching</p>
                    </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-200">
                    <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                        <Zap className="w-5 h-5 text-green-600" />
                    </div>
                    <div>
                        <p className="font-medium text-gray-900 text-sm">Consistent</p>
                        <p className="text-xs text-gray-500">Master design plan</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
