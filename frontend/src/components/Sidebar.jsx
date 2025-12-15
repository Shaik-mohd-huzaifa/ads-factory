import {
    Sparkles,
    LayoutGrid,
    Trash2,
    Palette,
    MessageSquare,
    HelpCircle,
    Plus,
    Shield // Added Shield for Brand Safe section
} from 'lucide-react'

export function Sidebar({
    activeView,
    setActiveView,
    brands,
    selectedBrand,
    setSelectedBrand,
    setShowBrandModal,
    className
}) {
    return (
        <aside className={`w-64 bg-white border-r border-gray-200 flex flex-col ${className}`}>
            {/* Logo */}
            <div className="p-4 border-b border-gray-200">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                        <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-xl text-gray-900">AdaptiveBrand</span>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4">
                <div className="space-y-1">
                    <button
                        onClick={() => setActiveView('campaigns')}
                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg font-medium ${activeView === 'campaigns' ? 'text-gray-900 bg-gray-100' : 'text-gray-600 hover:bg-gray-50'}`}
                    >
                        <LayoutGrid className="w-5 h-5" />
                        Campaigns
                    </button>
                    <button
                        onClick={() => setActiveView('brands')}
                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg font-medium ${activeView === 'brands' ? 'text-gray-900 bg-gray-100' : 'text-gray-600 hover:bg-gray-50'}`}
                    >
                        <Palette className="w-5 h-5" />
                        Brand Kits
                    </button>
                    <button className="w-full flex items-center gap-3 px-3 py-2 text-gray-600 hover:bg-gray-50 rounded-lg">
                        <Trash2 className="w-5 h-5" />
                        Recently deleted
                    </button>
                </div>

                <div className="mt-8">
                    <div className="flex items-center justify-between px-3 mb-2">
                        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Brands</p>
                        <button
                            onClick={() => setShowBrandModal(true)}
                            className="text-indigo-600 hover:text-indigo-700"
                        >
                            <Plus className="w-4 h-4" />
                        </button>
                    </div>
                    <div className="space-y-1">
                        {brands.map((brand) => (
                            <button
                                key={brand.id}
                                onClick={() => setSelectedBrand(brand)}
                                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${selectedBrand?.id === brand.id
                                        ? 'bg-indigo-50 text-indigo-700'
                                        : 'text-gray-600 hover:bg-gray-50'
                                    }`}
                            >
                                <div
                                    className="w-4 h-4 rounded"
                                    style={{ backgroundColor: brand.primary_color }}
                                />
                                {brand.name}
                            </button>
                        ))}
                        {brands.length === 0 && (
                            <p className="px-3 text-sm text-gray-400">No brands yet</p>
                        )}
                    </div>
                </div>
            </nav>

            {/* Bottom section */}
            <div className="p-4 border-t border-gray-200">
                <div className="bg-indigo-50 rounded-xl p-4 mb-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Shield className="w-5 h-5 text-indigo-600" />
                        <span className="font-medium text-gray-900">Brand Safe</span>
                    </div>
                    <p className="text-sm text-gray-600">All assets checked for compliance</p>
                </div>
                <div className="space-y-1">
                    <button className="w-full flex items-center gap-3 px-3 py-2 text-gray-600 hover:bg-gray-50 rounded-lg">
                        <MessageSquare className="w-5 h-5" />
                        Feedback
                    </button>
                    <button className="w-full flex items-center gap-3 px-3 py-2 text-gray-600 hover:bg-gray-50 rounded-lg">
                        <HelpCircle className="w-5 h-5" />
                        Help
                    </button>
                </div>
            </div>
        </aside>
    )
}
