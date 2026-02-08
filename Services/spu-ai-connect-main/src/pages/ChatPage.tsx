import { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Send,
  User,
  Copy,
  Check,
  Trash2,
  GraduationCap,
  DollarSign,
  ClipboardCheck,
  BookOpen,
  Phone,
  Scale,
  FileText,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import spuLogo from '@/assets/spu-logo.png';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { useLanguage } from '@/contexts/LanguageContext';
import { toast } from 'sonner';

interface Source {
  content: string;
  score: number;
  chunk_id?: string;
  metadata?: {
    faculty?: string;
    doc_category?: string;
  };
}

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: Source[];
}

interface QuickAction {
  key: string;
  icon: React.ReactNode;
  query: string;
}

const API_ENDPOINT = import.meta.env.VITE_API_URL || "http://localhost:5005/chat";
const API_STREAM_ENDPOINT = import.meta.env.VITE_API_STREAM_URL || "http://localhost:5005/chat/stream";

async function sendMessageToAPI(query: string, conversationId?: string) {
  const response = await fetch(API_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: query,
      k: 8,
      min_relevance_score: 0.3,
      conversation_id: conversationId || undefined
    })
  });

  const data = await response.json();
  return {
    answer: data.answer,
    conversationId: data.conversation_id,
    sources: data.sources,
    confidence: data.confidence,
    language: data.language
  };
}

/**
 * Stream message from API using Server-Sent Events
 * @param query - The user's question
 * @param conversationId - Optional conversation ID for context
 * @param onToken - Callback for each token received
 * @param onMetadata - Callback when metadata is received
 * @param onError - Callback for errors
 */
async function sendMessageStream(
  query: string,
  conversationId: string | undefined,
  onToken: (token: string) => void,
  onMetadata: (metadata: { conversationId: string; sources: unknown[]; confidence: number; language: string }) => void,
  onError: (error: string) => void
) {
  try {
    const response = await fetch(API_STREAM_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        k: 8,
        min_relevance_score: 0.3,
        conversation_id: conversationId || undefined
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('No response body');
    }

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events (lines ending with \n\n)
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'token') {
              onToken(data.content);
              // Add delay for typewriter effect - 25ms feels natural
              await new Promise(resolve => setTimeout(resolve, 25));
            } else if (data.type === 'metadata') {
              onMetadata({
                conversationId: data.conversation_id,
                sources: data.sources,
                confidence: data.confidence,
                language: data.language
              });
            } else if (data.type === 'error') {
              onError(data.content);
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

const ChatPage = () => {
  const { t } = useTranslation();
  const { isRTL } = useLanguage();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());

  const quickActions: QuickAction[] = [
    { key: 'faculties', icon: <GraduationCap className="h-5 w-5" />, query: isRTL ? 'ما هي الكليات المتاحة في الجامعة السورية الخاصة؟' : 'What faculties are available at SPU?' },
    { key: 'fees', icon: <DollarSign className="h-5 w-5" />, query: isRTL ? 'ما هي الرسوم الدراسية؟' : 'What are the tuition fees?' },
    { key: 'admission', icon: <ClipboardCheck className="h-5 w-5" />, query: isRTL ? 'ما هي شروط القبول في الجامعة؟' : 'What are the admission requirements?' },
    { key: 'programs', icon: <BookOpen className="h-5 w-5" />, query: isRTL ? 'ما هي البرامج والتخصصات المتاحة؟' : 'What programs and majors are available?' },
    { key: 'contact', icon: <Phone className="h-5 w-5" />, query: isRTL ? 'كيف يمكنني التواصل مع الجامعة؟' : 'How can I contact the university?' },
    { key: 'rules', icon: <Scale className="h-5 w-5" />, query: isRTL ? 'ما هي القواعد والقرارات في الجامعة؟' : 'What are the university rules and regulations?' },
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input.trim(),
      role: 'user',
      timestamp: new Date(),
    };

    const userQuery = input.trim();
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Create a placeholder message for streaming
    const aiMessageId = (Date.now() + 1).toString();
    let streamedContent = '';

    // Add empty assistant message that will be updated
    setMessages((prev) => [...prev, {
      id: aiMessageId,
      content: '',
      role: 'assistant',
      timestamp: new Date(),
    }]);

    try {
      await sendMessageStream(
        userQuery,
        conversationId,
        // onToken - update message content as tokens arrive
        (token) => {
          streamedContent += token;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? { ...msg, content: streamedContent }
                : msg
            )
          );
        },
        // onMetadata - update conversation ID and sources when received
        (metadata) => {
          if (metadata.conversationId) {
            setConversationId(metadata.conversationId);
          }
          // Store sources in the message
          if (metadata.sources && Array.isArray(metadata.sources)) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiMessageId
                  ? { ...msg, sources: metadata.sources as Source[] }
                  : msg
              )
            );
          }
        },
        // onError - show error in message
        (error) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId
                ? { ...msg, content: `Error: ${error}` }
                : msg
            )
          );
        }
      );
    } catch (error) {
      // Fallback for demo when API is not available
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMessageId
            ? {
              ...msg,
              content: isRTL
                ? 'شكراً لرسالتك! هذا رد تجريبي. سيتم ربط النظام بالخادم قريباً.\n\n**معلومات إضافية:**\n- يمكنك السؤال عن أي شيء\n- النظام يدعم العربية والإنجليزية'
                : 'Thank you for your message! This is a demo response. The system will be connected to the server soon.\n\n**Additional Information:**\n- You can ask about anything\n- The system supports Arabic and English',
            }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickAction = (query: string) => {
    setInput(query);
    textareaRef.current?.focus();
  };

  const handleCopy = async (content: string, id: string) => {
    await navigator.clipboard.writeText(content);
    setCopiedId(id);
    toast.success(t('chat.copied'));
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleClearChat = () => {
    setMessages([]);
    setConversationId(undefined);
    setExpandedSources(new Set());
    toast.success(isRTL ? 'تم مسح المحادثة' : 'Chat cleared');
  };

  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] animate-fade-in">
      {/* Hero Section - shown when no messages */}
      {!hasMessages && (
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-8">
          <div className="text-center mb-8 animate-fade-in">
            <h1 className="text-2xl md:text-4xl font-bold text-primary mb-4">
              {t('chat.welcome')}
            </h1>
            <p className="text-lg text-muted-foreground max-w-xl mx-auto">
              {t('chat.subtitle')}
            </p>
          </div>

          {/* Quick Actions Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4 max-w-2xl w-full">
            {quickActions.map((action, index) => (
              <button
                key={action.key}
                onClick={() => handleQuickAction(action.query)}
                className={cn(
                  "flex flex-col items-center gap-2 p-4 rounded-xl",
                  "bg-card border-2 border-border hover:border-primary",
                  "transition-all duration-300 hover:shadow-lg hover:scale-[1.02]",
                  "group animate-fade-in"
                )}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="p-3 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  {action.icon}
                </div>
                <span className="text-sm font-medium text-center text-foreground">
                  {t(`chat.quickActions.${action.key}`)}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages Container - shown when there are messages */}
      {hasMessages && (
        <div className="flex-1 overflow-y-auto space-y-4 pb-4 px-2 md:px-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                'flex gap-3 animate-fade-in',
                message.role === 'user'
                  ? (isRTL ? 'flex-row' : 'flex-row-reverse')
                  : (isRTL ? 'flex-row-reverse' : 'flex-row')
              )}
            >
              {/* Avatar */}
              <div
                className={cn(
                  'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium overflow-hidden',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                )}
              >
                {message.role === 'user' ? (
                  <User className="h-5 w-5" />
                ) : (
                  <img src={spuLogo} alt="SPU" className="h-8 w-8 object-contain" />
                )}
              </div>

              {/* Message Bubble */}
              <div className="max-w-[70%] md:max-w-[75%] group">
                <div
                  className={cn(
                    'px-4 py-3 relative',
                    message.role === 'user'
                      ? cn(
                        'bg-primary text-primary-foreground',
                        isRTL
                          ? 'rounded-tl-2xl rounded-tr-sm rounded-br-2xl rounded-bl-2xl'
                          : 'rounded-tr-2xl rounded-tl-sm rounded-bl-2xl rounded-br-2xl'
                      )
                      : cn(
                        'bg-muted text-foreground',
                        isRTL
                          ? 'rounded-tr-2xl rounded-tl-sm rounded-bl-2xl rounded-br-2xl'
                          : 'rounded-tl-2xl rounded-tr-sm rounded-br-2xl rounded-bl-2xl'
                      )
                  )}
                >
                  {message.role === 'assistant' ? (
                    <div className="prose prose-sm dark:prose-invert max-w-none prose-p:leading-loose prose-p:mb-6 prose-li:mb-2">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="leading-relaxed whitespace-pre-wrap">{message.content}</p>
                  )}

                </div>

                <div className={cn(
                  "flex items-center gap-2 mt-1",
                  message.role === 'user' ? (isRTL ? 'justify-start' : 'justify-end') : (isRTL ? 'justify-end' : 'justify-start')
                )}>
                  {/* Copy button for assistant messages */}
                  {message.role === 'assistant' && (
                    <button
                      onClick={() => handleCopy(message.content, message.id)}
                      className={cn(
                        "opacity-0 group-hover:opacity-100 transition-opacity",
                        "p-1 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground",
                      )}
                      title={t('chat.copy')}
                    >
                      {copiedId === message.id ? (
                        <Check className="h-3.5 w-3.5 text-green-500" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </button>
                  )}

                  {/* View Sources button for assistant messages */}
                  {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                    <button
                      onClick={() => toggleSources(message.id)}
                      className={cn(
                        "flex items-center gap-1 px-2 py-1 rounded-md text-xs",
                        "bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                      )}
                    >
                      <FileText className="h-3 w-3" />
                      <span>{isRTL ? `عرض المصادر (${message.sources.length})` : `View Sources (${message.sources.length})`}</span>
                      {expandedSources.has(message.id) ? (
                        <ChevronUp className="h-3 w-3" />
                      ) : (
                        <ChevronDown className="h-3 w-3" />
                      )}
                    </button>
                  )}

                  <span className="text-xs text-muted-foreground">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>

                {/* Sources Panel */}
                {message.role === 'assistant' && message.sources && expandedSources.has(message.id) && (
                  <div className="mt-3 space-y-2 animate-fade-in">
                    <div className="text-xs font-medium text-muted-foreground mb-2">
                      {isRTL ? 'المصادر المستخدمة:' : 'Sources Used:'}
                    </div>
                    {message.sources.map((source, idx) => (
                      <div
                        key={idx}
                        className={cn(
                          "p-3 rounded-lg border border-border bg-background/50",
                          "text-sm"
                        )}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-primary">
                            {isRTL ? `المصدر ${idx + 1}` : `Source ${idx + 1}`}
                          </span>
                          <span className={cn(
                            "text-xs px-2 py-0.5 rounded-full",
                            source.score >= 0.7 ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" :
                              source.score >= 0.4 ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400" :
                                "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                          )}>
                            {(source.score * 100).toFixed(0)}% {isRTL ? 'تطابق' : 'match'}
                          </span>
                        </div>
                        {source.metadata && (source.metadata.faculty || source.metadata.doc_category) && (
                          <div className="flex gap-2 mb-2 flex-wrap">
                            {source.metadata.faculty && (
                              <span className="text-xs px-2 py-0.5 rounded bg-muted">
                                {source.metadata.faculty}
                              </span>
                            )}
                            {source.metadata.doc_category && (
                              <span className="text-xs px-2 py-0.5 rounded bg-muted">
                                {source.metadata.doc_category}
                              </span>
                            )}
                          </div>
                        )}
                        <p className={cn(
                          "text-muted-foreground text-xs leading-relaxed",
                          isRTL && "text-right"
                        )}>
                          {source.content.length > 300
                            ? source.content.substring(0, 300) + '...'
                            : source.content}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {isLoading && (
            <div className={cn(
              'flex gap-3',
              isRTL ? 'flex-row-reverse' : 'flex-row'
            )}>
              <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center overflow-hidden">
                <img src={spuLogo} alt="SPU" className="h-8 w-8 object-contain" />
              </div>
              <div className="bg-muted px-4 py-3 rounded-2xl">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* Input Area */}
      <div className="mt-auto border-t border-border bg-background/80 backdrop-blur-sm p-4 shadow-[0_-4px_20px_-5px_rgba(0,0,0,0.1)]">
        <div className="flex items-end gap-3 max-w-4xl mx-auto">
          {/* Clear chat button */}
          {hasMessages && (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearChat}
              className="shrink-0 text-muted-foreground hover:text-destructive"
              title={t('chat.clearChat')}
            >
              <Trash2 className="h-5 w-5" />
            </Button>
          )}

          {/* Textarea */}
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t('chat.placeholder')}
              className={cn(
                "min-h-[48px] max-h-[150px] resize-none rounded-xl border-2",
                "border-border focus:border-primary transition-colors",
                "pr-4 pl-4 py-3",
                isRTL && "text-right"
              )}
              disabled={isLoading}
              rows={1}
            />
            {/* Character counter */}
            {input.length > 100 && (
              <span className={cn(
                "absolute bottom-1 text-xs text-muted-foreground",
                isRTL ? 'left-3' : 'right-3'
              )}>
                {input.length}
              </span>
            )}
          </div>

          {/* Send button */}
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="shrink-0 h-12 px-5 rounded-xl bg-primary hover:bg-primary/90 transition-all duration-300 shadow-soft hover:shadow-glow"
          >
            <Send className={cn("h-5 w-5", isRTL && "rotate-180")} />
            <span className="sr-only">{t('chat.send')}</span>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
