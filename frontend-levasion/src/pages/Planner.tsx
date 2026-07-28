import { useStreamingPlan } from '../hooks/useStreamingPlan';
import { useNavigate } from 'react-router-dom';

export default function Planner() {
  const { start, tokens, isStreaming, error, reasoningSteps } = useStreamingPlan();
  const navigate = useNavigate();

  const handleSearch = () => {
    start({ user_query: "Plan a luxury trip to Paris" });
  };

  return (
    <div className="p-8 max-w-5xl mx-auto flex flex-col md:flex-row gap-8 h-full">
      <div className="flex-1 space-y-6">
        <h2 className="text-3xl font-light">Planner</h2>
        <div className="bg-zinc-900 border border-gray-800 rounded-2xl p-6 space-y-4">
          <textarea 
            className="w-full bg-black border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 focus:outline-none focus:border-amber-500"
            rows={4}
            placeholder="Tell me your dream destination..."
          />
          <button 
            onClick={handleSearch}
            className="w-full bg-amber-500 hover:bg-amber-400 text-black py-3 rounded-xl font-medium transition-colors"
          >
            Generate Itinerary
          </button>
        </div>
        
        {error && (
          <div className="bg-red-900/50 border border-red-800 text-red-200 p-4 rounded-xl">
            {error}
          </div>
        )}

        <div className="bg-zinc-900 border border-gray-800 rounded-2xl p-6 min-h-[300px]">
          <h3 className="text-xl mb-4 text-amber-500">Concierge Insights</h3>
          <div className="text-gray-300 whitespace-pre-wrap">{tokens || "Awaiting instructions..."}</div>
          {isStreaming && <span className="animate-pulse">_</span>}
          
          {(tokens || error) && !isStreaming && (
            <button
              onClick={() => navigate('/checkout')}
              className="mt-6 bg-amber-500 hover:bg-amber-400 text-black px-6 py-2 rounded-full font-medium transition-colors"
            >
              Book Flight
            </button>
          )}
        </div>
      </div>
      
      <div className="w-full md:w-80 space-y-6">
        <div className="bg-zinc-900 border border-gray-800 rounded-2xl p-6 h-full">
          <h3 className="text-xl mb-4 text-amber-500">Curated Segments</h3>
          <ul className="space-y-3">
            {reasoningSteps.map((step, i) => (
              <li key={i} className="text-sm text-gray-400 border-l-2 border-amber-500 pl-3">
                {step}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
