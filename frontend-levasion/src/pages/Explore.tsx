
import { Link, useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';

export default function Explore() {
  const navigate = useNavigate();
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 space-y-8">
      <div className="text-center space-y-4 max-w-2xl">
        <h2 className="text-5xl font-light tracking-tight">Where will you escape to?</h2>
        <p className="text-gray-400 text-lg">Experience frictionless travel planning with your AI concierge.</p>
      </div>
      <div className="w-full max-w-3xl relative">
        <input 
          type="text" 
          placeholder="e.g. Find me a cheap flight from Delhi to Mumbai next Friday..." 
          className="w-full bg-zinc-900 border border-gray-700 rounded-full py-4 px-8 text-lg text-white focus:outline-none focus:border-amber-500 transition-colors placeholder-gray-500"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              navigate('/planner');
            }
          }}
        />
        <Link to="/planner" className="absolute right-2 top-2 bottom-2 bg-amber-500 hover:bg-amber-400 text-black rounded-full px-6 flex items-center justify-center font-medium transition-colors">
          <Search size={20} className="mr-2" />
          Plan Trip
        </Link>
      </div>
    </div>
  );
}
