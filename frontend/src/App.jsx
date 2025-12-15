import { useState, useEffect } from 'react'
import { Search, Instagram, Youtube, Linkedin, Twitter, Zap, FileText } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
import { BrandModal } from './components/BrandModal'
import { RecentCampaigns } from './components/RecentCampaigns'
import { CampaignResults } from './components/CampaignResults'
import { CampaignGenerator } from './components/CampaignGenerator'
import { TemplatesView } from './components/TemplatesView'
import { UploadView } from './components/UploadView'

const API_BASE = 'http://localhost:8001'

const PLATFORMS = [
  { id: 'instagram_feed', name: 'Instagram Feed', icon: Instagram, ratio: '1:1', color: 'bg-pink-500' },
  { id: 'instagram_story', name: 'Instagram Story', icon: Instagram, ratio: '9:16', color: 'bg-purple-500' },
  { id: 'facebook_feed', name: 'Facebook', icon: FileText, ratio: '1.91:1', color: 'bg-blue-600' },
  { id: 'youtube_thumbnail', name: 'YouTube', icon: Youtube, ratio: '16:9', color: 'bg-red-500' },
  { id: 'linkedin', name: 'LinkedIn', icon: Linkedin, ratio: '1.91:1', color: 'bg-blue-700' },
  { id: 'twitter', name: 'Twitter/X', icon: Twitter, ratio: '16:9', color: 'bg-gray-800' },
  { id: 'tiktok', name: 'TikTok', icon: Zap, ratio: '9:16', color: 'bg-black' },
]

const SAMPLE_CAMPAIGNS = [
  { id: 1, title: "Summer Sale 2024", brand: "TechCorp", platforms: 4, time: "2 days ago", color: "bg-blue-500" },
  { id: 2, title: "Product Launch", brand: "StyleBrand", platforms: 6, time: "3 days ago", color: "bg-purple-500" },
  { id: 3, title: "Holiday Special", brand: "FoodCo", platforms: 5, time: "1 week ago", color: "bg-amber-500" },
  { id: 4, title: "Brand Awareness", brand: "StartupXYZ", platforms: 7, time: "2 weeks ago", color: "bg-emerald-500" },
]

function App() {
  const [prompt, setPrompt] = useState('')
  const [activeTab, setActiveTab] = useState('generate')
  const [activeView, setActiveView] = useState('campaigns')
  const [isGenerating, setIsGenerating] = useState(false)
  const [selectedPlatforms, setSelectedPlatforms] = useState(['instagram_feed', 'instagram_story', 'facebook_feed', 'youtube_thumbnail'])
  const [brands, setBrands] = useState([])
  const [selectedBrand, setSelectedBrand] = useState(null)
  const [showBrandModal, setShowBrandModal] = useState(false)
  const [newBrand, setNewBrand] = useState({ name: '', primary_color: '#6366f1', industry: 'technology' })
  const [generatedCampaign, setGeneratedCampaign] = useState(null)
  const [campaignHistory, setCampaignHistory] = useState([])
  const [uploadedAssetIds, setUploadedAssetIds] = useState([])  // Track uploaded asset IDs for generation

  useEffect(() => {
    fetchBrands()
  }, [])

  const fetchBrands = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/brands`)
      const data = await res.json()
      setBrands(data)
      if (data.length > 0 && !selectedBrand) {
        setSelectedBrand(data[0])
      }
    } catch (e) {
      console.error('Failed to fetch brands:', e)
      // Fallback for demo when backend is not running
      if (brands.length === 0) setBrands([
        { id: 1, name: 'Demo Brand', primary_color: '#6366f1', industry: 'technology' }
      ])
    }
  }

  const createBrand = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/brands`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newBrand)
      })
      const data = await res.json()
      setBrands([...brands, data])
      setSelectedBrand(data)
      setShowBrandModal(false)
      setNewBrand({ name: '', primary_color: '#6366f1', industry: 'technology' })
    } catch (e) {
      console.error('Failed to create brand:', e)
      // Fallback for demo
      const mockBrand = { ...newBrand, id: Date.now() }
      setBrands([...brands, mockBrand])
      setSelectedBrand(mockBrand)
      setShowBrandModal(false)
    }
  }

  const togglePlatform = (platformId) => {
    setSelectedPlatforms(prev =>
      prev.includes(platformId)
        ? prev.filter(p => p !== platformId)
        : [...prev, platformId]
    )
  }

  const handleGenerate = async () => {
    if (!prompt.trim() || !selectedBrand) return
    setIsGenerating(true)
    setGeneratedCampaign(null)

    try {
      const res = await fetch(`${API_BASE}/api/campaigns/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          brand_id: selectedBrand.id,
          prompt,
          platforms: selectedPlatforms,
          asset_ids: uploadedAssetIds,  // Pass uploaded asset IDs for reference
        })
      })
      const data = await res.json()
      setGeneratedCampaign(data)
      setCampaignHistory(prev => [data, ...prev])
    } catch (e) {
      console.error('Failed to generate campaign:', e)
      // Demo fallback
      setTimeout(() => {
        const mockCampaign = {
          id: Date.now(),
          prompt,
          brief: {
            headline: "Future of Tech is Here",
            tagline: "Experience the revolution",
            suggested_cta: "Shop Now"
          },
          creatives: selectedPlatforms.map(pid => ({
            platform: pid,
            format: { ratio: PLATFORMS.find(p => p.id === pid).ratio },
            status: 'completed',
            // No image_url simulates generating state or placeholder
          }))
        }
        setGeneratedCampaign(mockCampaign)
        setCampaignHistory(prev => [mockCampaign, ...prev])
        setIsGenerating(false)
      }, 2000)
    } finally {
      if (!generatedCampaign) setIsGenerating(false) // Only if not handled by fallback
    }
  }

  const startNewCampaign = () => {
    setPrompt('')
    setGeneratedCampaign(null)
  }

  const renderContent = () => {
    if (activeView === 'brands') {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Brand management coming soon...
        </div>
      )
    }

    if (activeTab === 'templates') {
      return <TemplatesView />
    }

    if (activeTab === 'upload') {
      return <UploadView />
    }

    // Default 'generate' tab content
    return (
      <>
        <CampaignGenerator
          prompt={prompt}
          setPrompt={setPrompt}
          selectedBrand={selectedBrand}
          handleGenerate={handleGenerate}
          isGenerating={isGenerating}
          platforms={PLATFORMS}
          selectedPlatforms={selectedPlatforms}
          togglePlatform={togglePlatform}
          setShowBrandModal={setShowBrandModal}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onAssetsChange={setUploadedAssetIds}
        />

        {generatedCampaign ? (
          <CampaignResults
            campaign={generatedCampaign}
            platforms={PLATFORMS}
            onNewCampaign={startNewCampaign}
          />
        ) : (
          <RecentCampaigns
            history={campaignHistory}
            sampleCampaigns={SAMPLE_CAMPAIGNS}
            onSelectCampaign={setGeneratedCampaign}
          />
        )}
      </>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar
        activeView={activeView}
        setActiveView={setActiveView}
        brands={brands}
        selectedBrand={selectedBrand}
        setSelectedBrand={setSelectedBrand}
        setShowBrandModal={setShowBrandModal}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">
            {activeView === 'campaigns' ? 'Campaign Studio' : 'Brand Kits'}
          </h1>
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search campaigns..."
                className="pl-10 pr-4 py-2 bg-gray-100 border-0 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 w-64"
              />
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center text-white font-medium text-sm">
                U
              </div>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-auto p-6">
          {renderContent()}
        </div>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 px-6 py-3 text-center text-sm text-gray-500">
          AdaptiveBrand Studio • Sketch & Search Hackathon 2024
        </footer>
      </main>

      <BrandModal
        show={showBrandModal}
        onClose={() => setShowBrandModal(false)}
        onCreate={createBrand}
        newBrand={newBrand}
        setNewBrand={setNewBrand}
      />
    </div>
  )
}

export default App
