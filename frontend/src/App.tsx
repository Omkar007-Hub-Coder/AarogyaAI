import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import ProfileForm from './pages/ProfileForm'
import Recommendations from './pages/Recommendations'
import Dashboard from './pages/Dashboard'
import PoseRecognition from './pages/PoseRecognition'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-8 max-w-4xl">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/profile" element={<ProfileForm />} />
          {/* Legacy route — kept for existing tests */}
          <Route path="/recommendations/:userId" element={<Recommendations />} />
          {/* New personalization dashboard */}
          <Route path="/dashboard" element={<Dashboard />} />
          {/* Pose recognition */}
          <Route path="/pose" element={<PoseRecognition />} />
        </Routes>
      </main>
    </div>
  )
}
