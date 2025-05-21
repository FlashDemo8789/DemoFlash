import React, { useState } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  CardActionArea,
  Avatar,
  Paper,
  Stack,
  useTheme,
  alpha,
  Divider,
  Chip
} from '@mui/material';
import { 
  TrendingUp as TrendingUpIcon,
  Lightbulb as LightbulbIcon,
  Public as PublicIcon,
  People as PeopleIcon,
  ArrowForward as ArrowForwardIcon,
  QueryStats as QueryStatsIcon,
  Bookmark as BookmarkIcon,
  BarChart as BarChartIcon,
  FileDownload as FileDownloadIcon,
  AutoAwesome as AutoAwesomeIcon,
  BoltOutlined as FlashIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const HomePage: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  
  // Sample metrics data
  const sampleMetrics = [
    { name: 'Capital', icon: <TrendingUpIcon />, color: '#0066FF', description: 'Runway, burn rate, funding efficiency' },
    { name: 'Advantage', icon: <LightbulbIcon />, color: '#7C3AED', description: 'IP, technology moat, competitive edge' },
    { name: 'Market', icon: <PublicIcon />, color: '#F43F5E', description: 'TAM, growth potential, market share' },
    { name: 'People', icon: <PeopleIcon />, color: '#10B981', description: 'Team strength, hiring, culture' },
  ];
  
  // Features
  const features = [
    { 
      title: 'AI-Powered Analysis', 
      icon: <QueryStatsIcon sx={{ fontSize: 32, color: '#3B82F6' }} />,
      description: 'Our ML algorithms analyze 100+ metrics to predict startup success probability'
    },
    { 
      title: 'Investment Portfolio', 
      icon: <BookmarkIcon sx={{ fontSize: 32, color: '#7C3AED' }} />,
      description: 'Track and compare startups in your portfolio with custom dashboards'
    },
    { 
      title: 'Benchmark Reports', 
      icon: <BarChartIcon sx={{ fontSize: 32, color: '#F43F5E' }} />,
      description: 'Compare performance against similar startups in your sector'
    },
    { 
      title: 'PDF Exports', 
      icon: <FileDownloadIcon sx={{ fontSize: 32, color: '#10B981' }} />,
      description: 'Generate professional reports for team and investor sharing'
    },
  ];
  
  // Testimonials
  const testimonials = [
    {
      quote: "FlashCAMP helped us optimize our capital strategy and extend runway by 40%.",
      author: "Sarah Johnson",
      role: "Founder, TechFlow",
      avatar: "https://randomuser.me/api/portraits/women/44.jpg"
    },
    {
      quote: "The CAMP analysis gave us clear insights into our competitive advantage gaps.",
      author: "Michael Chen",
      role: "CEO, NexusAI",
      avatar: "https://randomuser.me/api/portraits/men/32.jpg"
    }
  ];

  return (
    <Box sx={{ 
      bgcolor: '#F9FAFC', 
      minHeight: '100vh',
      pt: { xs: 2, md: 0 } 
    }}>
      {/* Hero Section */}
      <Box 
        sx={{ 
          background: 'linear-gradient(135deg, #2E3192 0%, #4A38BD 100%)',
          pt: { xs: 8, md: 10 },
          pb: { xs: 10, md: 12 },
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {/* Decorative elements */}
        <Box sx={{
          position: 'absolute',
          top: -100,
          right: -100,
          width: 400,
          height: 400,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%)',
        }} />
        
        <Box sx={{
          position: 'absolute',
          bottom: -50,
          left: -50,
          width: 250,
          height: 250,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0) 70%)',
        }} />
      
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
              >
                <Typography 
                  variant="h2" 
                  component="h1" 
                  sx={{ 
                    color: 'white',
                    fontWeight: 800,
                    fontSize: { xs: '2.5rem', md: '3.5rem' },
                    lineHeight: 1.2,
                    mb: 2
                  }}
                >
                  Optimize Your <Box component="span" sx={{ color: '#8CEBFF' }}>Startup</Box> For Success
                </Typography>
              </motion.div>
              
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.1 }}
              >
                <Typography 
                  variant="h6" 
                  sx={{ 
                    color: 'rgba(255,255,255,0.8)', 
                    mb: 4,
                    fontWeight: 400,
                    fontSize: { xs: '1rem', md: '1.15rem' },
                    maxWidth: 520
                  }}
                >
                  FlashCAMP analyzes your Capital, Advantage, Market, and People to predict success probability and identify optimization opportunities.
                </Typography>
              </motion.div>
              
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                <Button 
                  variant="contained" 
                  size="large"
                  onClick={() => navigate('/analyze')}
                  endIcon={<ArrowForwardIcon />}
                  sx={{ 
                    px: 4, 
                    py: 1.5, 
                    borderRadius: '12px',
                    backgroundImage: 'linear-gradient(135deg, #21D4FD 0%, #2152FF 100%)',
                    boxShadow: '0 10px 20px rgba(33, 82, 255, 0.3)',
                    fontWeight: 600,
                    fontSize: '1.1rem',
                    textTransform: 'none',
                    '&:hover': {
                      backgroundImage: 'linear-gradient(135deg, #21D4FD 20%, #2152FF 100%)',
                      boxShadow: '0 15px 25px rgba(33, 82, 255, 0.35)',
                      transform: 'translateY(-2px)'
                    },
                    transition: 'all 0.3s ease'
                  }}
                >
                  Start Your Analysis
                </Button>
              </motion.div>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.3 }}
              >
                <Box
                  component="img"
                  src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjUwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8IS0tIEJhY2tncm91bmQgLS0+CiAgPHJlY3QgeD0iMCIgeT0iMCIgd2lkdGg9IjgwMCIgaGVpZ2h0PSI1MDAiIGZpbGw9IiMxZTI5M2IiIHJ4PSIxMCIgcnk9IjEwIi8+CiAgCiAgPCEtLSBIZWFkZXIgLS0+CiAgPHJlY3QgeD0iMjAiIHk9IjIwIiB3aWR0aD0iNzYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjMTExODI3IiByeD0iOCIgcnk9IjgiLz4KICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSIxNSIgZmlsbD0iIzNiODJmNiIvPgogIDx0ZXh0IHg9IjgwIiB5PSI1NSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjIwIiBmaWxsPSJ3aGl0ZSI+Rmxhc2hETkEgRGFzaGJvYXJkPC90ZXh0PgogIAogIDwhLS0gTWFpbiBDb250ZW50IC0tPgogIDwhLS0gQ0FNUCBTY29yZSBDYXJkcyAtLT4KICA8cmVjdCB4PSIyMCIgeT0iMTAwIiB3aWR0aD0iMTcwIiBoZWlnaHQ9IjEyMCIgZmlsbD0iIzExMTgyNyIgcng9IjgiIHJ5PSI4Ii8+CiAgPHRleHQgeD0iNDAiIHk9IjEzMCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE2IiBmaWxsPSJ3aGl0ZSI+Q2FwaXRhbDwvdGV4dD4KICA8dGV4dCB4PSI0MCIgeT0iMTcwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMzAiIGZpbGw9IiMwMDk0ZmYiPjc4JTwvdGV4dD4KICAKICAKICA8cmVjdCB4PSIyMTAiIHk9IjEwMCIgd2lkdGg9IjE3MCIgaGVpZ2h0PSIxMjAiIGZpbGw9IiMxMTE4MjciIHJ4PSI4IiByeT0iOCIvPgogIDx0ZXh0IHg9IjIzMCIgeT0iMTMwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IndoaXRlIj5BZHZhbnRhZ2U8L3RleHQ+CiAgPHRleHQgeD0iMjMwIiB5PSIxNzAiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIzMCIgZmlsbD0iIzdjM2FlZCI+NjUlPC90ZXh0PgogIAogIDxyZWN0IHg9IjQwMCIgeT0iMTAwIiB3aWR0aD0iMTcwIiBoZWlnaHQ9IjEyMCIgZmlsbD0iIzExMTgyNyIgcng9IjgiIHJ5PSI4Ii8+CiAgPHRleHQgeD0iNDIwIiB5PSIxMzAiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0id2hpdGUiPk1hcmtldDwvdGV4dD4KICA8dGV4dCB4PSI0MjAiIHk9IjE3MCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjMwIiBmaWxsPSIjZjQzZjVlIj44MiU8L3RleHQ+CiAgCiAgPHJlY3QgeD0iNTkwIiB5PSIxMDAiIHdpZHRoPSIxNzAiIGhlaWdodD0iMTIwIiBmaWxsPSIjMTExODI3IiByeD0iOCIgcnk9IjgiLz4KICA8dGV4dCB4PSI2MTAiIHk9IjEzMCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE2IiBmaWxsPSJ3aGl0ZSI+UGVvcGxlPC90ZXh0PgogIDx0ZXh0IHg9IjYxMCIgeT0iMTcwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMzAiIGZpbGw9IiMxMGI5ODEiPjcxJTwvdGV4dD4KCiAgPCEtLSBDaGFydCAtLT4KICA8cmVjdCB4PSIyMCIgeT0iMjQwIiB3aWR0aD0iNDYwIiBoZWlnaHQ9IjI0MCIgZmlsbD0iIzExMTgyNyIgcng9IjgiIHJ5PSI4Ii8+CiAgPHRleHQgeD0iNDAiIHk9IjI3MCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE2IiBmaWxsPSJ3aGl0ZSI+UGVyZm9ybWFuY2UgVHJlbmRzPC90ZXh0PgogIAogIDwhLS0gU2ltcGxpZmllZCBjaGFydCAtLT4KICA8cG9seWxpbmUgcG9pbnRzPSI0MCwzODAgMTIwLDM0MCAxODAsMzgwIDI0MCwzMDAgMzAwLDMyMCAzNjAsI2I5ICA0MjAsMzAwIiBzdHJva2U9IiMzYjgyZjYiIHN0cm9rZS13aWR0aD0iMyIgZmlsbD0ibm9uZSIvPgogIDxsaW5lIHgxPSI0MCIgeTE9IjQyMCIgeDI9IjQ0MCIgeTI9IjQyMCIgc3Ryb2tlPSIjNjQ3NDhiIiBzdHJva2Utd2lkdGg9IjIiLz4KICA8bGluZSB4MT0iNDAiIHkxPSIzMDAiIHgyPSI0MCIgeTI9IjQyMCIgc3Ryb2tlPSIjNjQ3NDhiIiBzdHJva2Utd2lkdGg9IjIiLz4KICAKICAKICA8IS0tIFNpZGViYXIgLS0+CiAgPHJlY3QgeD0iNTAwIiB5PSIyNDAiIHdpZHRoPSIyNjAiIGhlaWdodD0iMjQwIiBmaWxsPSIjMTExODI3IiByeD0iOCIgcnk9IjgiLz4KICA8dGV4dCB4PSI1MjAiIHk9IjI3MCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE2IiBmaWxsPSJ3aGl0ZSI+T3ZlcmFsbCBTY29yZTwvdGV4dD4KICA8Y2lyY2xlIGN4PSI2MzAiIGN5PSIzNTAiIHI9IjcwIiBmaWxsPSJub25lIiBzdHJva2U9IiM2NDc0OGIiIHN0cm9rZS13aWR0aD0iNCIvPgogIDxjaXJjbGUgY3g9IjYzMCIgY3k9IjM1MCIgcj0iNzAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzNiODJmNiIgc3Ryb2tlLXdpZHRoPSI4IiBzdHJva2UtZGFzaGFycmF5PSIyODAgMTIwIiBzdHJva2UtZGFzaG9mZnNldD0iMCIvPgogIDx0ZXh0IHg9IjYzMCIgeT0iMzU1IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMzAiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj43NCU8L3RleHQ+CiAgPHRleHQgeD0iNjMwIiB5PSIzODAiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzY0NzQ4YiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Rmxhc2hETkEgU2NvcmU8L3RleHQ+Cjwvc3ZnPg=="
                  alt="FlashCAMP Dashboard Preview"
                  sx={{
                    width: '100%',
                    maxWidth: 550,
                    height: 'auto',
                    borderRadius: '12px',
                    boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
                    transform: 'perspective(1000px) rotateY(-5deg) rotateX(5deg)',
                    mx: 'auto',
                    display: { xs: 'none', md: 'block' }
                  }}
                />
              </motion.div>
            </Grid>
          </Grid>
          
          {/* CAMP Metrics Preview */}
          <Box sx={{ mt: { xs: 6, md: 8 } }}>
            <Grid container spacing={2}>
              {sampleMetrics.map((metric, index) => (
                <Grid item xs={6} md={3} key={metric.name}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 + (index * 0.1) }}
                  >
                    <Paper
                      elevation={0}
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        borderRadius: '12px',
                        backgroundColor: 'rgba(255,255,255,0.1)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          backgroundColor: 'rgba(255,255,255,0.15)',
                          transform: 'translateY(-5px)'
                        }
                      }}
                    >
                      <Avatar
                        sx={{
                          bgcolor: `${alpha(metric.color, 0.2)}`,
                          color: metric.color,
                          width: 50,
                          height: 50,
                          mx: 'auto',
                          mb: 1
                        }}
                      >
                        {metric.icon}
                      </Avatar>
                      <Typography variant="h6" sx={{ color: 'white', fontWeight: 600, mb: 0.5 }}>
                        {metric.name}
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.8rem' }}>
                        {metric.description}
                      </Typography>
                    </Paper>
                  </motion.div>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Container>
      </Box>
      
      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: { xs: 6, md: 10 } }}>
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography 
            variant="h3" 
            component="h2" 
            sx={{ 
              fontWeight: 800, 
              mb: 2,
              fontSize: { xs: '2rem', md: '2.5rem' },
              background: 'linear-gradient(135deg, #3B3EAE 0%, #4A46D8 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Supercharge Your Startup Analysis
          </Typography>
          <Typography variant="h6" sx={{ color: 'text.secondary', maxWidth: 700, mx: 'auto', mb: 4 }}>
            Our comprehensive tools and metrics help you make data-driven decisions
          </Typography>
        </Box>
        
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={feature.title}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 + (index * 0.1) }}
              >
                <Card 
                  elevation={0} 
                  sx={{ 
                    height: '100%', 
                    borderRadius: '16px',
                    border: `1px solid ${theme.palette.grey[100]}`,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      boxShadow: '0 10px 30px rgba(0,0,0,0.08)',
                      transform: 'translateY(-5px)'
                    }
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ mb: 2 }}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h6" component="h3" sx={{ fontWeight: 700, mb: 1 }}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.9rem' }}>
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Container>
      
      {/* Testimonials */}
      <Box sx={{ backgroundColor: alpha(theme.palette.primary.main, 0.04), py: { xs: 6, md: 10 } }}>
        <Container maxWidth="lg">
          <Typography 
            variant="h4" 
            component="h2" 
            sx={{ 
              fontWeight: 700, 
              mb: 6, 
              textAlign: 'center',
              color: theme.palette.text.primary
            }}
          >
            What Our Users Say
          </Typography>
          
          <Grid container spacing={4} justifyContent="center">
            {testimonials.map((testimonial, index) => (
              <Grid item xs={12} md={6} key={testimonial.author}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 + (index * 0.1) }}
                >
                  <Paper
                    elevation={0}
                    sx={{
                      p: 4,
                      borderRadius: '16px',
                      border: `1px solid ${theme.palette.grey[100]}`,
                      height: '100%',
                      position: 'relative',
                      backgroundColor: '#FFFFFF',
                      '&:before': {
                        content: '""',
                        position: 'absolute',
                        top: 24,
                        left: 24,
                        width: 40,
                        height: 40,
                        backgroundImage: 'url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 24 24\' fill=\'%234A38BD\' opacity=\'0.2\'><path d=\'M6.5 10c-.223 0-.437.034-.65.065.069-.232.14-.468.254-.68.114-.308.292-.575.469-.844.148-.291.409-.488.601-.737.201-.242.475-.403.692-.604.213-.21.492-.315.714-.463.232-.133.434-.28.65-.35.208-.086.39-.16.539-.222.302-.125.474-.197.474-.197L9.758 4.03c0 0-.218.052-.597.144C8.97 4.222 8.737 4.278 8.472 4.345c-.271.069-.56.164-.882.306-.317.135-.638.301-.932.51-.3.2-.592.46-.859.736-.287.281-.53.605-.736.951-.209.343-.345.711-.44 1.083-.107.373-.161.76-.161 1.166 0 .43.056.826.161 1.2.086.37.238.705.393 1.038.18.336.395.659.647.951.229.296.485.559.769.786.28.25.582.44.901.6.336.159.656.296.98.364.53.162 1.043.246 1.543.246 1.704 0 3.334-.739 4.5-2.031a6.92 6.92 0 0 0 1.302-2.096c.312-.744.5-1.56.5-2.413 0-.863-.177-1.685-.478-2.45-.238-.604-.563-1.163-.947-1.668-.428-.568-.916-1.077-1.463-1.51-.557-.442-1.157-.818-1.806-1.101a11.74 11.74 0 0 0-2.13-.614 11.733 11.733 0 0 0-2.256-.198c-1.543 0-3.013.358-4.37.99a11.71 11.71 0 0 0-3.62 2.6A11.827 11.827 0 0 0 1.2 8.903 11.79 11.79 0 0 0 .289 13.41c0 1.545.37 3.012.99 4.325.195.416.435.816.693 1.197.257.382.516.757.837 1.085.317.336.65.643 1.008.91.34.278.696.531 1.07.745a11.642 11.642 0 0 0 3.788 1.271c.409.068.844.1 1.278.12.432 0 .881-.045 1.313-.128a11.619 11.619 0 0 0 6.25-2.954l-3.343-3.34A4.452 4.452 0 0 1 11.5 17.5a4.55 4.55 0 0 1-4.54-4.29c.022 0 .045-.01.068-.01 2.312 0 4.21-1.82 4.21-4.08 0-.708-.204-1.373-.583-1.94a4.257 4.257 0 0 0-1.418-1.4A4.352 4.352 0 0 0 6.5 5a4.42 4.42 0 0 0-1.767.37 4.369 4.369 0 0 0-2.445 2.453A4.299 4.299 0 0 0 2 9.79 4.427 4.427 0 0 0 6.5 14c0 0-.334 0-.65-.065z\'/></svg>")',
                        backgroundRepeat: 'no-repeat',
                        opacity: 0.1
                      }
                    }}
                  >
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        fontStyle: 'italic', 
                        mb: 3, 
                        color: theme.palette.text.primary,
                        fontWeight: 500,
                        pl: 5
                      }}
                    >
                      "{testimonial.quote}"
                    </Typography>
                    
                    <Divider sx={{ mb: 3 }} />
                    
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Avatar 
                        src={testimonial.avatar} 
                        alt={testimonial.author}
                        sx={{ width: 50, height: 50 }}
                      />
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {testimonial.author}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {testimonial.role}
                        </Typography>
                      </Box>
                    </Stack>
                  </Paper>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>
      
      {/* Call to Action */}
      <Container maxWidth="md" sx={{ py: { xs: 6, md: 10 }, textAlign: 'center' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Paper
            elevation={0}
            sx={{
              p: { xs: 4, md: 6 },
              borderRadius: '20px',
              background: 'linear-gradient(135deg, #4A38BD 0%, #2E3192 100%)',
              boxShadow: '0 20px 40px rgba(46, 49, 146, 0.3)',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            {/* Decorative elements */}
            <Box sx={{
              position: 'absolute',
              top: -30,
              right: -30,
              width: 150,
              height: 150,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%)',
            }} />
            
            <Box sx={{
              position: 'absolute',
              bottom: -20,
              left: -20,
              width: 100,
              height: 100,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%)',
            }} />
            
            <Typography 
              variant="h3" 
              sx={{ 
                color: 'white', 
                fontWeight: 800, 
                mb: 2,
                fontSize: { xs: '1.8rem', md: '2.5rem' }
              }}
            >
              Ready to Optimize Your Startup?
            </Typography>
            
            <Typography 
              variant="h6" 
              sx={{ 
                color: 'rgba(255,255,255,0.8)', 
                mb: 4, 
                maxWidth: 600, 
                mx: 'auto',
                fontWeight: 400
              }}
            >
              Get your CAMP analysis in minutes and discover your startup's strengths and opportunities.
            </Typography>
            
            <Button 
              variant="contained" 
              size="large"
              onClick={() => navigate('/analyze')}
              endIcon={<AutoAwesomeIcon />}
              sx={{ 
                px: 4, 
                py: 1.5, 
                borderRadius: '12px',
                bgcolor: 'white',
                color: '#4A38BD',
                fontWeight: 600,
                fontSize: '1.1rem',
                textTransform: 'none',
                boxShadow: '0 10px 20px rgba(255, 255, 255, 0.2)',
                '&:hover': {
                  bgcolor: 'rgba(255, 255, 255, 0.9)',
                  boxShadow: '0 15px 25px rgba(255, 255, 255, 0.25)',
                },
                transition: 'all 0.3s ease'
              }}
            >
              Start Free Analysis
            </Button>
          </Paper>
        </motion.div>
      </Container>
      
      {/* Footer */}
      <Box 
        sx={{ 
          borderTop: `1px solid ${theme.palette.divider}`,
          py: 4, 
          bgcolor: 'background.paper'
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                <Box sx={{ 
                  width: 32, 
                  height: 32, 
                  borderRadius: '8px', 
                  bgcolor: theme.palette.primary.main,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <FlashIcon sx={{ color: 'white', fontSize: 20 }} />
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>
                  Flash DNA
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Helping startups optimize for success with data-driven analysis and insights.
              </Typography>
              <Typography variant="caption" color="text.secondary">
                © {new Date().getFullYear()} FlashCAMP. All rights reserved.
              </Typography>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>
                Product
              </Typography>
              <Stack spacing={1}>
                {['Features', 'Pricing', 'Case Studies', 'Testimonials'].map(item => (
                  <Typography 
                    key={item} 
                    variant="body2" 
                    component="a" 
                    href="#" 
                    sx={{ 
                      color: 'text.secondary',
                      textDecoration: 'none',
                      '&:hover': { color: theme.palette.primary.main }
                    }}
                  >
                    {item}
                  </Typography>
                ))}
              </Stack>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>
                Company
              </Typography>
              <Stack spacing={1}>
                {['About', 'Team', 'Careers', 'Contact Us'].map(item => (
                  <Typography 
                    key={item} 
                    variant="body2" 
                    component="a" 
                    href="#" 
                    sx={{ 
                      color: 'text.secondary',
                      textDecoration: 'none',
                      '&:hover': { color: theme.palette.primary.main }
                    }}
                  >
                    {item}
                  </Typography>
                ))}
              </Stack>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>
                Resources
              </Typography>
              <Stack spacing={1}>
                {['Blog', 'Help Center', 'Documentation', 'API'].map(item => (
                  <Typography 
                    key={item} 
                    variant="body2" 
                    component="a" 
                    href="#" 
                    sx={{ 
                      color: 'text.secondary',
                      textDecoration: 'none',
                      '&:hover': { color: theme.palette.primary.main }
                    }}
                  >
                    {item}
                  </Typography>
                ))}
              </Stack>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>
                Legal
              </Typography>
              <Stack spacing={1}>
                {['Terms', 'Privacy', 'Cookies', 'Licenses'].map(item => (
                  <Typography 
                    key={item} 
                    variant="body2" 
                    component="a" 
                    href="#" 
                    sx={{ 
                      color: 'text.secondary',
                      textDecoration: 'none',
                      '&:hover': { color: theme.palette.primary.main }
                    }}
                  >
                    {item}
                  </Typography>
                ))}
              </Stack>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
};

export default HomePage; 