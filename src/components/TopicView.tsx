import { useState, useMemo, useEffect, useRef } from 'react';
import { Topic, TopicProgress, QuizScore, GamificationData, ConsumableInventory } from '../types';
import { BookOpen, Code, HelpCircle, Brain, BarChart3, Play, ChevronDown, ChevronUp, CheckCircle, Clock, BookmarkIcon, Award } from 'lucide-react';
import Quiz from './Quiz';
import BiasVarianceDemo from './BiasVarianceDemo';
import { getQuizQuestionsForTopic } from '../data/quizQuestions';
import { selectRandomQuestions } from '../utils/quizUtils';
import { calculateTopicStatus } from '../utils/statusCalculation';
import { calculateDaysUntilDecay } from '../utils/decaySystem';

// Declare KaTeX renderMathInElement type
declare global {
  interface Window {
    renderMathInElement?: (element: HTMLElement, options?: any) => void;
  }
}

interface TopicViewProps {
  topic: Topic;
  userProgress: TopicProgress | undefined;
  gamificationData: GamificationData;
  onProgressUpdate: (progress: TopicProgress) => void;
  onAwardXP: (amount: number, reason: string) => void;
  onQuizComplete: (score: QuizScore) => void;
  onUseConsumable: (itemType: keyof ConsumableInventory) => void;
}

export default function TopicView({ topic, userProgress, gamificationData, onProgressUpdate, onQuizComplete, onUseConsumable }: TopicViewProps) {
  const [activeTab, setActiveTab] = useState<'theory' | 'code' | 'questions' | 'demo' | 'quiz'>('theory');
  const [showQuiz, setShowQuiz] = useState(false);
  const [expandedQuestions, setExpandedQuestions] = useState<Set<number>>(new Set());
  const theoryContentRef = useRef<HTMLDivElement>(null);
  
  // Get full question pool for this topic
  const questionPool = useMemo(() => {
    // Try to get from separate question files first
    const poolQuestions = getQuizQuestionsForTopic(topic.id);
    // Fall back to questions in topic definition if no pool exists
    return poolQuestions.length > 0 ? poolQuestions : (topic.quizQuestions || []);
  }, [topic.id, topic.quizQuestions]);

  // Select random questions when quiz starts (regenerate on each quiz start)
  const [quizQuestions, setQuizQuestions] = useState(() => selectRandomQuestions(questionPool, 10));

  // Reset quiz state when topic changes
  useEffect(() => {
    setShowQuiz(false);
    setActiveTab('theory');
    setQuizQuestions(selectRandomQuestions(questionPool, 10));
  }, [topic.id, questionPool]);

  // Trigger KaTeX rendering when content changes or theory tab is shown
  useEffect(() => {
    if (activeTab === 'theory' && theoryContentRef.current && window.renderMathInElement) {
      try {
        window.renderMathInElement(theoryContentRef.current, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
          ],
          throwOnError: false
        });
      } catch (err) {
        console.error('KaTeX rendering failed:', err);
      }
    }
  }, [activeTab, topic.content]);

  const handleQuizComplete = (quizScore: QuizScore) => {
    const updatedScores = [...(userProgress?.quizScores || []), quizScore];
    
    // Automatically calculate status based on quiz performance
    const statusResult = calculateTopicStatus(
      userProgress?.status || 'not_started',
      updatedScores,
      {
        masteryStrength: userProgress?.masteryStrength,
        lastMasteredDate: userProgress?.lastMasteredDate,
        highScoreStreak: userProgress?.highScoreStreak,
      }
    );
    
    const newProgress: TopicProgress = {
      ...userProgress,
      status: statusResult.status,
      lastAccessed: new Date(),
      quizScores: updatedScores,
      firstCompletion: userProgress?.firstCompletion || new Date(),
      masteryStrength: statusResult.masteryStrength,
      highScoreStreak: statusResult.highScoreStreak,
      lastMasteredDate: statusResult.shouldUpdateMasteredDate 
        ? new Date() 
        : userProgress?.lastMasteredDate,
    };
    onProgressUpdate(newProgress);
    
    // Call parent handler for gamification tracking
    onQuizComplete(quizScore);
  };

  const handleQuizClose = () => {
    setShowQuiz(false);
  };

  const handleStartQuiz = () => {
    // Generate new random questions each time quiz starts
    setQuizQuestions(selectRandomQuestions(questionPool, 10));
    setShowQuiz(true);
  };

  const handleQuizRetake = () => {
    // Generate new random questions when retaking quiz
    setQuizQuestions(selectRandomQuestions(questionPool, 10));
  };

  const tabs = [
    { id: 'theory', label: 'Theory', icon: BookOpen, count: null },
    { id: 'code', label: 'Code Examples', icon: Code, count: topic.codeExamples?.length || 0 },
    { id: 'questions', label: 'Interview Questions', icon: HelpCircle, count: topic.interviewQuestions?.length || 0 },
    ...(topic.hasInteractiveDemo ? [{ id: 'demo', label: 'Interactive Demo', icon: Play, count: null }] : []),
    { id: 'quiz', label: 'Practice Quiz', icon: Brain, count: questionPool.length || 0 }
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
          {/* Read-only status badge */}
          <div 
            className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium ${
              userProgress?.status === 'mastered' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' :
              userProgress?.status === 'reviewing' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' :
              'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-400'
            }`}
            title="Status automatically updates based on your quiz performance"
          >
            {userProgress?.status === 'mastered' ? <CheckCircle className="w-4 h-4" /> :
             userProgress?.status === 'reviewing' ? <Clock className="w-4 h-4" /> :
             <BookmarkIcon className="w-4 h-4" />}
            <span>
              {userProgress?.status === 'mastered' ? 'Mastered' :
               userProgress?.status === 'reviewing' ? 'Reviewing' :
               'Not Started'}
            </span>
          </div>
          {userProgress?.quizScores && userProgress.quizScores.length > 0 && (
            <div className="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400">
              <BarChart3 className="w-4 h-4" />
              <span>
                Best: {Math.max(...userProgress.quizScores.map(s => Math.round((s.score / s.totalQuestions) * 100)))}%
              </span>
            </div>
          )}
          {userProgress?.status === 'mastered' && userProgress.masteryStrength !== undefined && (
            <div className="flex items-center space-x-1 text-sm text-blue-600 dark:text-blue-400">
              <CheckCircle className="w-4 h-4" />
              <span title="Mastery strength - higher values mean slower decay">
                Strength: {userProgress.masteryStrength}/100
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
          <div className="theory-content" ref={theoryContentRef}>
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
            {showQuiz && questionPool.length > 0 ? (
              <Quiz
                key={`${topic.id}-${quizQuestions[0]?.id || Date.now()}`}
                questions={quizQuestions}
                onComplete={handleQuizComplete}
                onClose={handleQuizClose}
                onRetake={handleQuizRetake}
                consumableInventory={gamificationData.consumableInventory}
                onUseConsumable={onUseConsumable}
              />
            ) : (
              <div className="text-center py-8">
                {questionPool.length > 0 ? (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400 mb-2">
                      Test your knowledge with a quiz from a pool of {questionPool.length} questions
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-500 mb-4">
                      Each quiz randomly selects {Math.min(10, questionPool.length)} questions
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
                    
                    {/* Mastery Progress Info */}
                    {userProgress?.status !== 'mastered' && userProgress?.quizScores && userProgress.quizScores.length > 0 && (
                      <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                        <div className="text-sm text-blue-900 dark:text-blue-100">
                          <p className="font-medium mb-1">🎯 Path to Mastery:</p>
                          <ul className="text-xs space-y-1 text-blue-700 dark:text-blue-300">
                            <li>• Complete at least 2 quizzes</li>
                            <li>• Achieve 80%+ average on your last 3 quizzes</li>
                            <li>• OR score 100% twice in a row</li>
                          </ul>
                        </div>
                      </div>
                    )}
                    
                    {userProgress?.status === 'mastered' && (
                      <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                        <div className="text-sm text-green-900 dark:text-green-100">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <p className="font-medium flex items-center gap-2">
                                <Award className="w-4 h-4" />
                                ✨ Topic Mastered!
                              </p>
                              <p className="text-xs text-green-700 dark:text-green-300 mt-1">
                                Keep practicing to maintain your mastery status
                              </p>
                            </div>
                            {userProgress.masteryStrength !== undefined && (
                              <div className="ml-4 text-right">
                                <div className="text-lg font-bold text-green-700 dark:text-green-300">
                                  {userProgress.masteryStrength}/100
                                </div>
                                <div className="text-xs text-green-600 dark:text-green-400">
                                  Mastery Strength
                                </div>
                              </div>
                            )}
                          </div>
                          {(() => {
                            const daysUntilDecay = calculateDaysUntilDecay(userProgress);
                            if (daysUntilDecay !== null) {
                              return (
                                <div className={`mt-3 pt-3 border-t ${
                                  daysUntilDecay <= 7 
                                    ? 'border-orange-300 dark:border-orange-700'
                                    : 'border-green-300 dark:border-green-700'
                                }`}>
                                  <div className="flex items-center justify-between text-xs">
                                    <span className={daysUntilDecay <= 7 ? 'text-orange-700 dark:text-orange-300' : 'text-green-700 dark:text-green-300'}>
                                      {daysUntilDecay === 0 && '⚠️ Needs review now'}
                                      {daysUntilDecay > 0 && daysUntilDecay <= 7 && `⏰ Review in ${daysUntilDecay} ${daysUntilDecay === 1 ? 'day' : 'days'}`}
                                      {daysUntilDecay > 7 && `✅ Review in ${daysUntilDecay} days`}
                                    </span>
                                    <span className="text-green-600 dark:text-green-400">
                                      {userProgress.highScoreStreak || 0}🔥 streak
                                    </span>
                                  </div>
                                  {daysUntilDecay <= 7 && daysUntilDecay > 0 && (
                                    <p className="mt-2 text-xs text-orange-700 dark:text-orange-300">
                                      💡 This topic will decay soon. Take a quiz to refresh your mastery!
                                    </p>
                                  )}
                                </div>
                              );
                            }
                            return null;
                          })()}
                        </div>
                      </div>
                    )}
                    
                    <button
                      onClick={handleStartQuiz}
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