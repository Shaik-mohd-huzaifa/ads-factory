import { Download, Plus, Check, Image, Database, Sparkles, Palette, FileText, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

export function CampaignResults({
    campaign,
    platforms,
    onNewCampaign
}) {
    const [showRetrievedAssets, setShowRetrievedAssets] = useState(true)
    const [showDesignPlan, setShowDesignPlan] = useState(false)
    
    if (!campaign) return null

    const relevantAssets = campaign.relevant_assets || []
    const designPlan = campaign.master_design?.design_plan || {}

    return (
        <div className="max-w-5xl mx-auto mb-8 space-y-6">
            {/* Retrieved Assets from Vector DB */}
            {relevantAssets.length > 0 && (
                <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-2xl border border-purple-200 shadow-sm overflow-hidden">
                    <button 
                        onClick={() => setShowRetrievedAssets(!showRetrievedAssets)}
                        className="w-full px-6 py-4 flex items-center justify-between hover:bg-purple-100/50 transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-purple-100 rounded-xl flex items-center justify-center">
                                <Database className="w-5 h-5 text-purple-600" />
                            </div>
                            <div className="text-left">
                                <h3 className="font-semibold text-gray-900">Retrieved from Vector DB</h3>
                                <p className="text-sm text-gray-500">{relevantAssets.length} relevant assets found</p>
                            </div>
                        </div>
                        {showRetrievedAssets ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
                    </button>
                    
                    {showRetrievedAssets && (
                        <div className="px-6 pb-6">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {relevantAssets.map((asset, idx) => (
                                    <div key={idx} className="bg-white rounded-xl p-4 border border-purple-100 hover:shadow-md transition-shadow">
                                        <div className="flex items-start gap-3">
                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                                                asset.type === 'design' ? 'bg-pink-100' : 
                                                asset.type === 'campaign' ? 'bg-blue-100' : 'bg-green-100'
                                            }`}>
                                                {asset.type === 'design' ? <Image className="w-4 h-4 text-pink-600" /> :
                                                 asset.type === 'campaign' ? <Sparkles className="w-4 h-4 text-blue-600" /> :
                                                 <FileText className="w-4 h-4 text-green-600" />}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 uppercase">
                                                        {asset.type || 'asset'}
                                                    </span>
                                                    <span className="text-xs text-purple-600 font-medium">
                                                        {(asset.score * 100).toFixed(0)}% match
                                                    </span>
                                                </div>
                                                <p className="text-sm font-medium text-gray-900 mt-1 truncate">
                                                    {asset.name || asset.headline || asset.prompt?.slice(0, 40) || 'Untitled'}
                                                </p>
                                                {asset.image_url && (
                                                    <div className="mt-2 rounded-lg overflow-hidden bg-gray-100 aspect-video">
                                                        <img src={asset.image_url} alt="" className="w-full h-full object-cover" />
                                                    </div>
                                                )}
                                                {asset.prompt && !asset.image_url && (
                                                    <p className="text-xs text-gray-500 mt-1 line-clamp-2">{asset.prompt}</p>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Design Plan */}
            {Object.keys(designPlan).length > 0 && (
                <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-2xl border border-amber-200 shadow-sm overflow-hidden">
                    <button 
                        onClick={() => setShowDesignPlan(!showDesignPlan)}
                        className="w-full px-6 py-4 flex items-center justify-between hover:bg-amber-100/50 transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-amber-100 rounded-xl flex items-center justify-center">
                                <Palette className="w-5 h-5 text-amber-600" />
                            </div>
                            <div className="text-left">
                                <h3 className="font-semibold text-gray-900">Master Design Plan</h3>
                                <p className="text-sm text-gray-500">Consistency blueprint for all platforms</p>
                            </div>
                        </div>
                        {showDesignPlan ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
                    </button>
                    
                    {showDesignPlan && (
                        <div className="px-6 pb-6">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {designPlan.hero_element && (
                                    <div className="bg-white rounded-xl p-3 border border-amber-100">
                                        <p className="text-xs font-medium text-amber-600 uppercase mb-1">Hero Element</p>
                                        <p className="text-sm text-gray-700">{designPlan.hero_element}</p>
                                    </div>
                                )}
                                {designPlan.mood && (
                                    <div className="bg-white rounded-xl p-3 border border-amber-100">
                                        <p className="text-xs font-medium text-amber-600 uppercase mb-1">Mood</p>
                                        <p className="text-sm text-gray-700">{designPlan.mood}</p>
                                    </div>
                                )}
                                {designPlan.lighting_style && (
                                    <div className="bg-white rounded-xl p-3 border border-amber-100">
                                        <p className="text-xs font-medium text-amber-600 uppercase mb-1">Lighting</p>
                                        <p className="text-sm text-gray-700">{designPlan.lighting_style}</p>
                                    </div>
                                )}
                                {designPlan.color_palette && (
                                    <div className="bg-white rounded-xl p-3 border border-amber-100">
                                        <p className="text-xs font-medium text-amber-600 uppercase mb-1">Colors</p>
                                        <p className="text-sm text-gray-700">{designPlan.color_palette}</p>
                                    </div>
                                )}
                            </div>
                            {designPlan.core_visual && (
                                <div className="mt-3 bg-white rounded-xl p-3 border border-amber-100">
                                    <p className="text-xs font-medium text-amber-600 uppercase mb-1">Core Visual</p>
                                    <p className="text-sm text-gray-700">{designPlan.core_visual}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Main Campaign Results */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h3 className="text-xl font-bold text-gray-900">Generated Campaign</h3>
                        <p className="text-sm text-gray-500 mt-1">{campaign.creatives?.length || 0} platform variants created</p>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="px-3 py-1.5 bg-green-100 text-green-700 rounded-full text-sm font-medium flex items-center gap-1">
                            <Check className="w-4 h-4" />
                            Brand Compliant
                        </span>
                    </div>
                </div>

                {/* Brief */}
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-5 mb-6 border border-indigo-100">
                    <h4 className="text-lg font-bold text-gray-900 mb-2">{campaign.brief?.headline}</h4>
                    <p className="text-gray-600 text-base">{campaign.brief?.tagline}</p>
                    <div className="flex items-center gap-4 mt-3">
                        <span className="px-3 py-1 bg-indigo-600 text-white rounded-lg text-sm font-medium">
                            {campaign.brief?.suggested_cta}
                        </span>
                        {campaign.brief?.mood && (
                            <span className="text-sm text-indigo-600">Mood: {campaign.brief.mood}</span>
                        )}
                    </div>
                </div>

                {/* Platform Previews */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {campaign.creatives?.map((creative, idx) => {
                        const platform = platforms.find(p => p.id === creative.platform)
                        return (
                            <div key={idx} className="relative group cursor-pointer">
                                {creative.image_url ? (
                                    <div className="rounded-xl overflow-hidden aspect-square bg-gray-100 ring-2 ring-transparent hover:ring-indigo-400 transition-all">
                                        <img
                                            src={creative.image_url}
                                            alt={platform?.name}
                                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                        />
                                    </div>
                                ) : (
                                    <div className={`${platform?.color || 'bg-gray-400'} rounded-xl aspect-square flex items-center justify-center`}>
                                        <Image className="w-10 h-10 text-white/50" />
                                    </div>
                                )}
                                <div className="mt-2 text-center">
                                    <p className="text-sm font-medium text-gray-700">{platform?.name}</p>
                                    <p className="text-xs text-gray-400">{creative.format?.ratio || platform?.ratio}</p>
                                </div>
                                {creative.is_master && (
                                    <div className="absolute top-2 left-2 px-2 py-0.5 bg-indigo-600 text-white text-xs rounded-full font-medium">
                                        Master
                                    </div>
                                )}
                                {creative.status === 'completed' && (
                                    <div className="absolute top-2 right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
                                        <Check className="w-4 h-4 text-white" />
                                    </div>
                                )}
                            </div>
                        )
                    })}
                </div>

                {/* Actions */}
                <div className="flex gap-3 mt-6">
                    <button className="flex-1 flex items-center justify-center gap-2 px-4 py-3.5 bg-gray-900 text-white rounded-xl font-medium hover:bg-gray-800 transition-colors">
                        <Download className="w-5 h-5" />
                        Export All
                    </button>
                    <button
                        onClick={onNewCampaign}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-3.5 bg-indigo-600 text-white rounded-xl font-medium hover:bg-indigo-700 transition-colors"
                    >
                        <Plus className="w-5 h-5" />
                        New Campaign
                    </button>
                </div>
            </div>
        </div>
    )
}
