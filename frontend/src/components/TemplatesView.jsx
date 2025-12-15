import { Layout } from 'lucide-react'

const MOCK_TEMPLATES = [
    { id: 1, name: "Product Launch", type: "Social Media", count: 7 },
    { id: 2, name: "Seasonal Sale", type: "E-commerce", count: 5 },
    { id: 3, name: "Event Promotion", type: "Events", count: 4 },
    { id: 4, name: "Brand Awareness", type: "Identity", count: 6 },
    { id: 5, name: "Newsletter", type: "Email", count: 3 },
    { id: 6, name: "App Download", type: "Mobile", count: 5 },
]

export function TemplatesView() {
    return (
        <div className="max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900">Recommended Templates</h2>
                <div className="flex gap-2">
                    <select className="px-3 py-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <option>All Types</option>
                        <option>Social Media</option>
                        <option>E-commerce</option>
                    </select>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {MOCK_TEMPLATES.map((template) => (
                    <div
                        key={template.id}
                        className="group cursor-pointer bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg transition-all hover:-translate-y-1"
                    >
                        <div className="h-40 bg-gray-100 flex items-center justify-center group-hover:bg-indigo-50 transition-colors">
                            <Layout className="w-12 h-12 text-gray-300 group-hover:text-indigo-400" />
                        </div>
                        <div className="p-4">
                            <h3 className="font-semibold text-gray-900 mb-1">{template.name}</h3>
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-gray-500">{template.type}</span>
                                <span className="text-xs font-medium text-indigo-600 bg-indigo-50 px-2 py-1 rounded">
                                    {template.count} formats
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
