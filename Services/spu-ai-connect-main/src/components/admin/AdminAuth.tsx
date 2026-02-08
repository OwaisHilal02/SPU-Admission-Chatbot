import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Lock, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

const ADMIN_PASSWORD = 'spu-admin-2026';
const AUTH_KEY = 'spu-admin-auth';

interface AdminAuthProps {
  children: React.ReactNode;
}

export const checkAdminAuth = (): boolean => {
  return sessionStorage.getItem(AUTH_KEY) === 'authenticated';
};

export const logoutAdmin = () => {
  sessionStorage.removeItem(AUTH_KEY);
};

const AdminAuth = ({ children }: AdminAuthProps) => {
  const { t } = useTranslation();
  const [isAuthenticated, setIsAuthenticated] = useState(checkAdminAuth());
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(false);
  const [isShaking, setIsShaking] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (password === ADMIN_PASSWORD) {
      sessionStorage.setItem(AUTH_KEY, 'authenticated');
      setIsAuthenticated(true);
      setError(false);
    } else {
      setError(true);
      setIsShaking(true);
      setTimeout(() => setIsShaking(false), 500);
    }
  };

  if (isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-[calc(100vh-12rem)] flex items-center justify-center p-4">
      <Card className={cn(
        "w-full max-w-md glass-card animate-fade-in",
        isShaking && "animate-shake"
      )}>
        <CardHeader className="text-center">
          <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
            <Lock className="h-8 w-8 text-primary" />
          </div>
          <CardTitle className="text-2xl">{t('admin.auth.title')}</CardTitle>
          <CardDescription>{t('admin.auth.subtitle')}</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <Input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(false);
                }}
                placeholder={t('admin.auth.password')}
                className={cn(
                  "h-12 pr-12",
                  error && "border-destructive focus:border-destructive"
                )}
                autoFocus
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              </button>
            </div>
            
            {error && (
              <p className="text-sm text-destructive animate-fade-in">
                {t('admin.auth.error')}
              </p>
            )}
            
            <Button 
              type="submit" 
              className="w-full h-12 bg-primary hover:bg-primary/90"
              disabled={!password.trim()}
            >
              {t('admin.auth.submit')}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default AdminAuth;
