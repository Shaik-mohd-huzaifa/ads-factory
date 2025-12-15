import { UploadCloud, File, X } from 'lucide-react'
import { useState } from 'react'

export function UploadView() {
    const [files, setFiles] = useState([])
    const [isDragging, setIsDragging] = useState(false)

    const handleDragOver = (e) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setIsDragging(false)
        const droppedFiles = Array.from(e.dataTransfer.files)
        setFiles(prev => [...prev, ...droppedFiles])
    }

    const handleFileInput = (e) => {
        const selectedFiles = Array.from(e.target.files)
        setFiles(prev => [...prev, ...selectedFiles])
    }

    const removeFile = (idx) => {
        setFiles(prev => prev.filter((_, i) => i !== idx))
    }

    return (
        <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
                <h2 className="text-xl font-semibold text-gray-900 mb-6 text-center">Upload Assets</h2>

                <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center transition-colors cursor-pointer ${isDragging
                            ? 'border-indigo-500 bg-indigo-50'
                            : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
                        }`}
                    onClick={() => document.getElementById('file-input').click()}
                >
                    <input
                        id="file-input"
                        type="file"
                        multiple
                        className="hidden"
                        onChange={handleFileInput}
                    />
                    <div className="w-16 h-16 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mb-4">
                        <UploadCloud className="w-8 h-8" />
                    </div>
                    <p className="text-lg font-medium text-gray-900 mb-2">
                        Click to upload or drag and drop
                    </p>
                    <p className="text-sm text-gray-500 text-center max-w-sm">
                        SVG, PNG, JPG or GIF (max. 800x400px)
                    </p>
                </div>

                {files.length > 0 && (
                    <div className="mt-8 space-y-3">
                        <h3 className="text-sm font-medium text-gray-700">Uploaded Files ({files.length})</h3>
                        {files.map((file, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 bg-white border border-gray-200 rounded flex items-center justify-center">
                                        <File className="w-5 h-5 text-gray-400" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-gray-900">{file.name}</p>
                                        <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => removeFile(idx)}
                                    className="text-gray-400 hover:text-red-500 p-1"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        ))}

                        <div className="flex justify-end mt-4">
                            <button className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors">
                                Process Files
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
