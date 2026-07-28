

export default function Profile() {
  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <h2 className="text-3xl font-light">Your Profile</h2>
      <div className="bg-zinc-900 border border-gray-800 rounded-2xl p-6">
        <h3 className="text-xl mb-4 text-amber-500">Active Bookings & Alerts</h3>
        <p className="text-gray-400">No active bookings currently. Start exploring to plan your next escape.</p>
      </div>
    </div>
  );
}
