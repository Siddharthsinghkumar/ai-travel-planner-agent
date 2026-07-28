import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Explore from './pages/Explore';
import Planner from './pages/Planner';
import Profile from './pages/Profile';
import Checkout from './pages/Checkout';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-black text-white font-sans flex flex-col">
        <header className="p-6 border-b border-gray-800 flex justify-between items-center bg-zinc-900">
          <h1 className="text-2xl font-light tracking-widest bg-gradient-to-r from-amber-200 to-amber-500 bg-clip-text text-transparent">L'ÉVASION</h1>
          <nav className="flex gap-8 text-sm uppercase tracking-wide">
            <Link to="/" className="hover:text-amber-400 transition-colors">Explore</Link>
            <Link to="/planner" className="hover:text-amber-400 transition-colors">Planner</Link>
            <Link to="/profile" className="hover:text-amber-400 transition-colors">Profile</Link>
          </nav>
        </header>
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Explore />} />
            <Route path="/planner" element={<Planner />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/checkout" element={<Checkout />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
