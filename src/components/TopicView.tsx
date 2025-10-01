import React, { useState } from 'react';
import { Topic, UserProgress, TopicProgress, QuizScore } from '../types';
import { BookOpen, Code, HelpCircle, Brain, BarChart3, Play, ChevronDown, ChevronUp } from 'lucide-react';
import Quiz from './Quiz';
import BiasVarianceDemo from './BiasVarianceDemo';

interface TopicViewProps {
  topic: Topic;
  userProgress: TopicProgress | undefined;
  onProgressUpdate: (progress: TopicProgress) => void;
}

export default function TopicView({ topic, userProgress, onProgressUpdate }: TopicViewProps) {
  const [activeTab, setActiveTab] = useState<'theory' | 'code' | 'questions' | 'demo' | 'quiz'>('theory');
  const [showQuiz, setShowQuiz] = useState(false);
  const [expandedQuestions, setExpandedQuestions] = useState<Set<number>>(new Set());

  const updateStatus = (status: 'not_started' | 'reviewing' | 'mastered') => {
    const newProgress: TopicProgress = {
      ...userProgress,
      status,
      lastAccessed: new Date(),
      quizScores: userProgress?.quizScores || []
    };
    onProgressUpdate(newProgress);
  };

  const handleQuizComplete = (quizScore: QuizScore) => {
    const newProgress: TopicProgress = {
      ...userProgress,
      status: userProgress?.status || 'reviewing',
      lastAccessed: new Date(),
      quizScores: [...(userProgress?.quizScores || []), quizScore]
    };
    onProgressUpdate(newProgress);
  };

  const handleQuizClose = () => {
    setShowQuiz(false);
  };

  const tabs = [
    { id: 'theory', label: 'Theory', icon: BookOpen, count: null },
    { id: 'code', label: 'Code Examples', icon: Code, count: topic.codeExamples?.length || 0 },
    { id: 'questions', label: 'Interview Questions', icon: HelpCircle, count: topic.interviewQuestions?.length || 0 },
    ...(topic.hasInteractiveDemo ? [{ id: 'demo', label: 'Interactive Demo', icon: Play, count: null }] : []),
    { id: 'quiz', label: 'Practice Quiz', icon: Brain, count: topic.quizQuestions?.length || 0 }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {topic.title}
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            {topic.description}
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={userProgress?.status || 'not_started'}
            onChange={(e) => updateStatus(e.target.value as any)}
            className="text-sm border border-gray-300 dark:border-gray-600 rounded-md px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="not_started">Not Started</option>
            <option value="reviewing">Reviewing</option>
            <option value="mastered">Mastered</option>
          </select>
          {userProgress?.quizScores && userProgress.quizScores.length > 0 && (
            <div className="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400">
              <BarChart3 className="w-4 h-4" />
              <span>
                Best: {Math.max(...userProgress.quizScores.map(s => Math.round((s.score / s.totalQuestions) * 100)))}%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex space-x-8 px-6">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-4 border-b-2 font-medium text-sm ${
                  isActive
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
                {tab.count !== null && (
                  <span className={`px-2 py-0.5 rounded-full text-xs ${
                    isActive
                      ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                      : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                  }`}>
                    {tab.count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'theory' && (
          <div className="theory-content">
            <div
              dangerouslySetInnerHTML={{ __html: topic.content }}
            />
          </div>
        )}

        {activeTab === 'code' && (
          <div className="space-y-6">
            {topic.codeExamples && topic.codeExamples.length > 0 ? (
              topic.codeExamples.map((example, index) => (
                <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                  <div className="bg-gray-50 dark:bg-gray-900 px-4 py-2 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {example.language}
                      </span>
                    </div>
                  </div>
                  <pre className="p-4 overflow-x-auto bg-gray-900 text-gray-100 text-sm">
                    <code>{example.code}</code>
                  </pre>
                  {example.explanation && (
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-sm text-blue-800 dark:text-blue-200">
                        {example.explanation}
                      </p>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                No code examples available for this topic yet.
              </div>
            )}
          </div>
        )}

        {activeTab === 'questions' && (
          <div className="space-y-4">
            {topic.interviewQuestions && topic.interviewQuestions.length > 0 ? (
              topic.interviewQuestions.map((item, index) => {
                const isExpanded = expandedQuestions.has(index);
                const toggleExpand = () => {
                  const newExpanded = new Set(expandedQuestions);
                  if (isExpanded) {
                    newExpanded.delete(index);
                  } else {
                    newExpanded.add(index);
                  }
                  setExpandedQuestions(newExpanded);
                };

                return (
                  <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                    <button
                      onClick={toggleExpand}
                      className="w-full p-4 flex items-start space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors text-left"
                    >
                      <span className="flex-shrink-0 w-6 h-6 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full flex items-center justify-center text-sm font-medium">
                        {index + 1}
                      </span>
                      <p className="flex-1 text-gray-700 dark:text-gray-300 font-medium">
                        {typeof item === 'string' ? item : item.question}
                      </p>
                      {typeof item === 'object' && item.answer && (
                        <span className="flex-shrink-0 text-gray-400 dark:text-gray-500">
                          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                        </span>
                      )}
                    </button>
                    {typeof item === 'object' && item.answer && isExpanded && (
                      <div className="px-4 pb-4 pt-2 bg-gray-50 dark:bg-gray-700/30 border-t border-gray-200 dark:border-gray-600">
                        <div className="pl-9">
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Answer:</p>
                          <div className="text-gray-700 dark:text-gray-300 prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap">
                            {item.answer}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                No interview questions available for this topic yet.
              </div>
            )}
          </div>
        )}

        {activeTab === 'demo' && topic.hasInteractiveDemo && (
          <div>
            {topic.id === 'bias-variance-tradeoff' && <BiasVarianceDemo />}
            {/* Add other interactive demos here as needed */}
          </div>
        )}

        {activeTab === 'quiz' && (
          <div>
            {showQuiz && topic.quizQuestions && topic.quizQuestions.length > 0 ? (
              <Quiz
                questions={topic.quizQuestions}
                onComplete={handleQuizComplete}
                onClose={handleQuizClose}
              />
            ) : (
              <div className="text-center py-8">
                {topic.quizQuestions && topic.quizQuestions.length > 0 ? (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      Test your knowledge with {topic.quizQuestions.length} practice questions
                    </p>
                    {userProgress?.quizScores && userProgress.quizScores.length > 0 && (
                      <div className="mb-6">
                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Previous Scores
                        </h4>
                        <div className="flex justify-center space-x-4">
                          {userProgress.quizScores.slice(-3).map((score, index) => (
                            <div key={index} className="text-center">
                              <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                                {Math.round((score.score / score.totalQuestions) * 100)}%
                              </div>
                              <div className="text-xs text-gray-500 dark:text-gray-400">
                                {score.date.toLocaleDateString()}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    <button
                      onClick={() => setShowQuiz(true)}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Start Quiz
                    </button>
                  </div>
                ) : (
                  <div className="text-gray-500 dark:text-gray-400">
                    No quiz questions available for this topic yet.
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}