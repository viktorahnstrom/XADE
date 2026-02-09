import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    KeyboardAvoidingView,
    Platform,
} from 'react-native';
import { colors } from '../theme/colors';

export default function LoginScreen() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = () => {
        console.log('Login pressed', { email, password });
    };

    const handleGoogleLogin = () => {
        console.log('Google login pressed');
    };

    return (
        <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
            <View style={styles.container}>
                {/* Logo */}
                <Text style={styles.logo}>XADE</Text>

                {/* Login Header */}
                <Text style={styles.header}>Login</Text>

                {/* Email Field */}
                <View style={styles.inputGroup}>
                    <Text style={styles.label}>Email</Text>
                    <TextInput style={styles.input} placeholder="you@example.com" placeholderTextColor="#999" value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize='none' />    
                </View>

                {/* Password Field */}
                <View style={styles.inputGroup}>
                    <Text style={styles.label}>Password</Text>
                    <TextInput style={styles.input} placeholder="Enter your password" placeholderTextColor="#999" value={password} onChangeText={setPassword} secureTextEntry />
                </View>

                {/* Links Row */}
                <View style={styles.linksRow}>
                    <TouchableOpacity>
                        <Text style={styles.link}>
                            New to XADE? <Text style={styles.linkBold}>Sign up</Text>
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity>
                        <Text style={styles.link}>Forgot your password?</Text>
                    </TouchableOpacity>
                </View>

                {/* Login Button */}
                <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
                    <Text style={styles.loginButtonText}>Login</Text>
                </TouchableOpacity>

                {/* Divider */}
                <View style={styles.divider}>
                    <View style={styles.dividerLine} />
                    <Text style={styles.dividerText}>or</Text>
                    <View style={styles.dividerLine} />
                </View>

                {/* Google Button */}
                <TouchableOpacity style={styles.googleButton} onPress={handleGoogleLogin}>
                    <Text style={styles.googleButtonText}>Continue with Google</Text>
                </TouchableOpacity>
            </View>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.pureWhite,
    },
    content: {
        flex: 1,
        paddingHorizontal: 24,
        paddingTop: 80,
    },
    logo: {
        fontSize: 48,
        fontWeight: '300',
        color: colors.deepBlue,
        textAlign: 'center',
        marginBottom: 40,
    },
    header: {
        fontSize: 28,
        fontWeight: '600',
        color: colors.charcoal,
        marginBottom: 24,
    },
    inputGroup: {
        marginBottom: 16,
    },
    label: {
        fontSize: 14,
        fontWeight: '500',
        color: colors.charcoal,
        marginBottom: 8,
    },
    input: {
        borderWidth: 1,
        borderColor: '#E0E0E0',
        borderRadius: 8,
        paddingHorizontal: 16,
        paddingVertical: 14,
        fontSize: 16,
        backgroundColor: colors.pureWhite,
        color: colors.charcoal,
    },
    linksRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginTop: 8,
        marginBottom: 24,
    },
    link: {
        fontSize: 13,
        color: colors.charcoal,
    },
    linkBold: {
        color: colors.deepBlue,
        fontWeight: '600',
    },
    loginButton: {
        backgroundColor: colors.deepBlue,
        borderRadius: 8,
        paddingVertical: 16,
        alignItems:'center',
    },
    loginButtonText: {
        color: colors.pureWhite,
        fontSize: 16,
        fontWeight: '600',
    },
    divider: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: 24,
    },
    dividerLine: {
        flex: 1,
        height: 1,
        backgroundColor: '#E0E0E0',
    },
    dividerText: {
        marginHorizontal: 16,
        color: '#999',
        fontSize: 14,
    },
    googleButton: {
        borderWidth: 1,
        borderColor: '#E0E0E0',
        borderRadius: 8,
        paddingVertical: 16,
        alignItems: 'center',
        backgroundColor: colors.pureWhite,
    },
    googleButtonText: {
        color: colors.charcoal,
        fontSize: 16,
        fontWeight: '500',
    },
});