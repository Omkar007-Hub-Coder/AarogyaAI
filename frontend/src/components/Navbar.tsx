import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()

  const links = [
    { to: '/',       label: 'Home' },
    { to: '/profile', label: 'Get Recommendations' },
    { to: '/pose',   label: 'Pose Recognition' },
  ]

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 max-w-4xl flex items-center justify-between h-14">
        <Link to="/" className="text-lg font-bold text-green-700 tracking-tight">
          🌿 AarogyaAI
        </Link>
        <div className="flex gap-1 text-sm">
          {links.map(({ to, label }) => (
            <Link key={to} to={to}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                pathname === to
                  ? 'bg-green-50 text-green-700 font-semibold'
                  : 'text-gray-600 hover:text-green-600 hover:bg-gray-50'
              }`}>
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
