import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="text-center py-16">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">
        Welcome to <span className="text-green-700">AarogyaAI</span>
      </h1>
      <p className="text-gray-600 text-lg mb-2 max-w-xl mx-auto">
        Personalized Yoga, Fitness &amp; Ahar recommendations powered by Machine Learning.
      </p>
      <p className="text-gray-500 text-sm mb-8 max-w-lg mx-auto">
        Enter your health profile and receive tailored suggestions for yoga poses, exercises,
        and a diet plan aligned with your goals.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-10 max-w-2xl mx-auto text-left">
        {[
          { emoji: '🧘', title: 'Yoga', desc: 'Poses matched to your flexibility and goals.' },
          { emoji: '💪', title: 'Fitness', desc: 'Exercises suited to your activity level.' },
          { emoji: '🥗', title: 'Ahar', desc: 'Diet recommendations for your body and goal.' },
        ].map(({ emoji, title, desc }) => (
          <div key={title} className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <div className="text-3xl mb-2">{emoji}</div>
            <h3 className="font-semibold text-gray-800 mb-1">{title}</h3>
            <p className="text-sm text-gray-500">{desc}</p>
          </div>
        ))}
      </div>

      <Link
        to="/profile"
        className="inline-block bg-green-600 hover:bg-green-700 text-white font-semibold px-8 py-3 rounded-lg transition-colors"
      >
        Start — Enter Your Profile
      </Link>
    </div>
  )
}
