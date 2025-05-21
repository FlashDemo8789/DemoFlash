import React, { useState } from "react";
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box,
  Container,
  IconButton,
  Menu,
  MenuItem,
  useMediaQuery,
  useTheme,
  Avatar,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from "@mui/material";
import { 
  Dashboard as DashboardIcon,
  BarChart as AnalyzeIcon,
  Business as PortfolioIcon,
  History as HistoryIcon,
  Person as ProfileIcon,
  Menu as MenuIcon,
  BoltOutlined as FlashIcon
} from '@mui/icons-material';
import { Link } from "react-router-dom";

const NavBar: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { name: "Dashboard", icon: <DashboardIcon />, path: "/" },
    { name: "Analyze", icon: <AnalyzeIcon />, path: "/analyze" },
    { name: "Portfolio", icon: <PortfolioIcon />, path: "/portfolio" },
    { name: "History", icon: <HistoryIcon />, path: "/history" },
    { name: "Profile", icon: <ProfileIcon />, path: "/profile" }
  ];

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  return (
    <>
      <AppBar 
        position="static" 
        elevation={0}
        sx={{ 
          background: 'linear-gradient(90deg, #2E3192 0%, #4A38BD 100%)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
          height: isMobile ? '70px' : '60px',
          justifyContent: 'center'
        }}
      >
        <Container maxWidth="xl">
          <Toolbar disableGutters sx={{ minHeight: isMobile ? '70px' : '60px' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: isMobile ? 1 : 0 }}>
              <FlashIcon sx={{ color: '#FFFFFF', mr: 1.5, fontSize: 24 }} />
              <Typography 
                variant="h6" 
                component={Link} 
                to="/"
                sx={{ 
                  flexGrow: 0, 
                  fontWeight: 700, 
                  color: '#FFFFFF', 
                  textDecoration: 'none',
                  letterSpacing: '0.5px',
                  fontSize: { xs: 18, md: 20 }
                }}
              >
                Flash DNA
              </Typography>
            </Box>

            {isMobile ? (
              <>
                <IconButton 
                  color="inherit" 
                  aria-label="open menu"
                  edge="end" 
                  onClick={toggleMobileMenu}
                  sx={{ color: '#FFFFFF' }}
                >
                  <MenuIcon />
                </IconButton>
                <Drawer
                  anchor="right"
                  open={mobileMenuOpen}
                  onClose={toggleMobileMenu}
                  sx={{
                    '& .MuiDrawer-paper': {
                      width: 240,
                      background: '#2F2F7E',
                      color: '#FFFFFF',
                      paddingTop: 1
                    }
                  }}
                >
                  <Box sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
                    <FlashIcon sx={{ mr: 1 }} />
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>
                      Flash DNA
                    </Typography>
                  </Box>
                  <Divider sx={{ backgroundColor: 'rgba(255,255,255,0.1)' }} />
                  <List sx={{ pt: 2 }}>
                    {navItems.map((item) => (
                      <ListItem 
                        key={item.name}
                        component={Link}
                        to={item.path}
                        button
                        onClick={toggleMobileMenu}
                        sx={{ 
                          py: 1.5,
                          '&:hover': { 
                            backgroundColor: 'rgba(255,255,255,0.1)'
                          }
                        }}
                      >
                        <ListItemIcon sx={{ color: '#FFFFFF', minWidth: 40 }}>
                          {item.icon}
                        </ListItemIcon>
                        <ListItemText 
                          primary={item.name} 
                          primaryTypographyProps={{ fontWeight: 500 }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Drawer>
              </>
            ) : (
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 4 }}>
                {navItems.map((item) => (
                  <Button 
                    key={item.name}
                    component={Link} 
                    to={item.path}
                    sx={{ 
                      color: '#FFFFFF', 
                      mx: 0.5,
                      px: 2,
                      py: 0.8,
                      fontSize: 14,
                      fontWeight: 500,
                      textTransform: 'none',
                      borderRadius: '8px',
                      '&:hover': {
                        backgroundColor: 'rgba(255,255,255,0.15)',
                      }
                    }}
                    startIcon={item.icon}
                  >
                    {item.name}
                  </Button>
                ))}
              </Box>
            )}
          </Toolbar>
        </Container>
      </AppBar>
    </>
  );
};

export default NavBar;
