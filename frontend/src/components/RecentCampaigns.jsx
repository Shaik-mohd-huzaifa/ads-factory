import { LayoutGrid } from 'lucide-react'

export function RecentCampaigns({
    history,
    sampleCampaigns,
    onSelectCampaign
}) {
    return (
        <div className="max-w-5xl mx-auto">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Recent Campaigns {history.length > 0 && `(${history.length})`}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {history.length > 0 ? (
                    history.map((campaign) => (
                        <div
                            key={campaign.id}
                            onClick={() => onSelectCampaign(campaign)}
                            className="bg-white rounded-xl overflow-hidden border border-gray-200 hover:shadow-lg transition-shadow cursor-pointer group"
                        >
                            {campaign.creatives?.[0]?.image_url ? (
                                <div className="h-32 relative">
                                    <img
                                        src={campaign.creatives[0].image_url}
                                        alt={campaign.prompt}
                                        className="w-full h-full object-cover"
                                    />
                                    <div className="absolute top-2 right-2 bg-white/90 text-gray-700 text-xs px-2 py-1 rounded font-medium">
                                        {campaign.creatives?.length || 0} formats
                                    </div>
                                </div>
                            ) : (
                                <div className="h-32 bg-indigo-500 relative flex items-center justify-center">
                                    <LayoutGrid className="w-10 h-10 text-white/50" />
                                    <div className="absolute top-2 right-2 bg-white/90 text-gray-700 text-xs px-2 py-1 rounded font-medium">
                                        {campaign.creatives?.length || 0} formats
                                    </div>
                                </div>
                            )}
                            <div className="p-4">
                                <h3 className="font-semibold text-gray-900 group-hover:text-indigo-600 transition-colors truncate">
                                    {campaign.brief?.headline || campaign.prompt?.slice(0, 30)}
                                </h3>
                                <div className="flex items-center justify-between mt-2">
                                    <span className="text-sm text-gray-500 truncate">{campaign.prompt?.slice(0, 20)}...</span>
                                    <span className="text-xs text-gray-400">Just now</span>
                                </div>
                            </div>
                        </div>
                    ))
                ) : (
                    sampleCampaigns.map((campaign) => (
                        <div
                            key={campaign.id}
                            className="bg-white rounded-xl overflow-hidden border border-gray-200 hover:shadow-lg transition-shadow cursor-pointer group opacity-50"
                        >
                            <div className={`h-32 ${campaign.color} relative flex items-center justify-center`}>
                                <LayoutGrid className="w-10 h-10 text-white/50" />
                                <div className="absolute top-2 right-2 bg-white/90 text-gray-700 text-xs px-2 py-1 rounded font-medium">
                                    {campaign.platforms} formats
                                </div>
                            </div>
                            <div className="p-4">
                                <h3 className="font-semibold text-gray-900 group-hover:text-indigo-600 transition-colors">
                                    {campaign.title}
                                </h3>
                                <div className="flex items-center justify-between mt-2">
                                    <span className="text-sm text-gray-500">{campaign.brand}</span>
                                    <span className="text-xs text-gray-400">{campaign.time}</span>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}
