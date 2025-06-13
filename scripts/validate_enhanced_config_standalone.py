#!/usr/bin/env python3
"""
Script de Validação de Configurações Enhanced Consolidadas v4.9.8
================================================================

Valida se todas as configurações enhanced consolidadas estão funcionando corretamente,
testando o carregamento de configurações por stage e inicialização dos componentes.

🔧 CONSOLIDAÇÃO: Validação do sistema unificado (base.py + cost_monitor.py)
✅ TESTES: Carregamento de configs, inicialização de componentes, fallbacks
🎯 MONITORAMENTO: Custos e performance por stage
📦 ENHANCED: Sistema enhanced integrado nos arquivos originais
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Encontrar raiz do projeto
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

def test_enhanced_loader():
    """Testa o carregamento do EnhancedConfigLoader consolidado"""
    logger.info("🧪 Testando EnhancedConfigLoader consolidado...")
    
    try:
        from anthropic_integration.base import get_enhanced_config_loader, load_operation_config
        
        # Testar singleton
        loader1 = get_enhanced_config_loader()
        loader2 = get_enhanced_config_loader()
        
        if loader1 is not loader2:
            logger.error("❌ Singleton pattern falhou")
            return False
            
        logger.info("✅ Singleton pattern funcionando")
        
        # Testar carregamento de stages
        test_operations = [
            'political_analysis',
            'sentiment_analysis', 
            'network_analysis',
            'qualitative_analysis',
            'pipeline_review',
            'topic_interpretation',
            'validation'
        ]
        
        for operation in test_operations:
            try:
                config = load_operation_config(operation)
                if 'model' not in config:
                    logger.error(f"❌ Configuração inválida para {operation}: falta 'model'")
                    return False
                logger.info(f"✅ {operation}: {config.get('model', 'N/A')}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar config para {operation}: {e}")
                return False
        
        logger.info("✅ EnhancedConfigLoader consolidado validado com sucesso")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        return False

def test_anthropic_base():
    """Testa inicialização do AnthropicBase com enhanced config"""
    logger.info("🧪 Testando AnthropicBase com enhanced config...")
    
    try:
        from anthropic_integration.base import AnthropicBase
        
        # Testar inicialização sem stage_operation
        base1 = AnthropicBase()
        logger.info(f"✅ AnthropicBase sem stage: {getattr(base1, 'model', 'N/A')}")
        
        # Testar inicialização com stage_operation
        base2 = AnthropicBase(stage_operation="political_analysis")
        logger.info(f"✅ AnthropicBase com political_analysis: {getattr(base2, 'model', 'N/A')}")
        
        # Verificar se enhanced config foi carregada
        if hasattr(base2, 'enhanced_config') and base2.enhanced_config:
            logger.info("✅ Enhanced config carregada com sucesso")
        else:
            logger.warning("⚠️ Enhanced config não carregada (usando fallback)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar AnthropicBase: {e}")
        return False

def test_component_initialization():
    """Testa inicialização dos componentes principais"""
    logger.info("🧪 Testando inicialização de componentes...")
    
    test_config = {
        'anthropic': {
            'api_key': '${ANTHROPIC_API_KEY}',
            'model': 'claude-3-5-sonnet-20241022'
        }
    }
    
    components_to_test = [
        ('PoliticalAnalyzer', 'political_analyzer'),
        ('AnthropicSentimentAnalyzer', 'sentiment_analyzer'),
        ('SmartPipelineReviewer', 'smart_pipeline_reviewer'),
        ('TopicInterpreter', 'topic_interpreter'),
        ('CompletePipelineValidator', 'pipeline_validator'),
        ('QualitativeClassifier', 'qualitative_classifier'),
        ('IntelligentNetworkAnalyzer', 'intelligent_network_analyzer')
    ]
    
    success_count = 0
    
    for class_name, module_name in components_to_test:
        try:
            module = __import__(f'anthropic_integration.{module_name}', fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Inicializar componente
            if class_name == 'CompletePipelineValidator':
                component = component_class(test_config, str(project_root))
            else:
                component = component_class(test_config)
            
            # Verificar se enhanced config foi aplicada
            model = getattr(component, 'model', 'N/A')
            logger.info(f"✅ {class_name}: {model}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar {class_name}: {e}")
    
    logger.info(f"✅ {success_count}/{len(components_to_test)} componentes inicializados com sucesso")
    return success_count == len(components_to_test)

def test_cost_monitor():
    """Testa o sistema de monitoramento de custos consolidado"""
    logger.info("🧪 Testando sistema de monitoramento de custos...")
    
    try:
        from anthropic_integration.cost_monitor import get_cost_monitor
        
        # Inicializar cost monitor
        monitor = get_cost_monitor(project_root)
        
        # Testar registro de uso
        cost = monitor.record_usage(
            model="claude-3-5-sonnet-20241022",
            input_tokens=100,
            output_tokens=50,
            stage="test_stage",
            operation="validation"
        )
        
        if cost > 0:
            logger.info(f"✅ Custo calculado: ${cost:.6f}")
        else:
            logger.warning("⚠️ Custo calculado como 0")
        
        # Testar relatório
        report = monitor.get_daily_report()
        logger.info(f"✅ Relatório diário gerado: {report.get('total_cost', 0):.6f} USD")
        
        # Testar auto-downgrade
        should_downgrade = monitor.should_auto_downgrade()
        logger.info(f"✅ Auto-downgrade: {'Ativo' if should_downgrade else 'Inativo'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar cost monitor: {e}")
        return False

def test_fallback_strategies():
    """Testa estratégias de fallback"""
    logger.info("🧪 Testando estratégias de fallback...")
    
    try:
        from anthropic_integration.base import get_enhanced_config_loader
        
        loader = get_enhanced_config_loader()
        
        # Testar fallbacks para diferentes modelos
        test_models = [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022", 
            "claude-3-5-haiku-20241022"
        ]
        
        for model in test_models:
            fallbacks = loader.get_fallback_models(model)
            if fallbacks:
                logger.info(f"✅ Fallbacks para {model}: {', '.join(fallbacks)}")
            else:
                logger.warning(f"⚠️ Nenhum fallback para {model}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar fallbacks: {e}")
        return False

def test_performance_modes():
    """Testa modos de performance"""
    logger.info("🧪 Testando modos de performance...")
    
    try:
        from anthropic_integration.base import get_enhanced_config_loader
        
        loader = get_enhanced_config_loader()
        
        # Testar diferentes operações (já que performance modes específicos não estão implementados)
        test_operations = ['political_analysis', 'sentiment_analysis', 'network_analysis']
        
        for operation in test_operations:
            config = loader.get_stage_config(loader.get_stage_from_operation(operation))
            model = config.get('model', 'N/A')
            logger.info(f"✅ Operação {operation}: {model}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar configurações por operação: {e}")
        return False

def run_validation_suite():
    """Executa suite completa de validação"""
    logger.info("🚀 Iniciando validação completa do sistema enhanced...")
    
    tests = [
        ("Enhanced Loader", test_enhanced_loader),
        ("Anthropic Base", test_anthropic_base),
        ("Component Initialization", test_component_initialization),
        ("Cost Monitor", test_cost_monitor),
        ("Fallback Strategies", test_fallback_strategies),
        ("Configuration per Operation", test_performance_modes)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 TESTE: {test_name}")
        logger.info('='*50)
        
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSOU")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FALHOU")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERRO - {e}")
    
    # Relatório final
    logger.info(f"\n{'='*50}")
    logger.info("📊 RELATÓRIO FINAL DE VALIDAÇÃO")
    logger.info('='*50)
    logger.info(f"✅ Testes passaram: {passed}/{total}")
    logger.info(f"❌ Testes falharam: {total - passed}/{total}")
    logger.info(f"📈 Taxa de sucesso: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 TODAS AS VALIDAÇÕES PASSARAM!")
        logger.info("✅ Sistema enhanced está funcionando corretamente")
        logger.info("")
        logger.info("📋 Próximos passos:")
        logger.info("1. Execute o pipeline: poetry run python run_pipeline.py")
        logger.info("2. Monitore logs para enhanced config loading")
        logger.info("3. Verifique relatórios de custo em logs/")
        return True
    else:
        logger.error("⚠️ ALGUMAS VALIDAÇÕES FALHARAM!")
        logger.error("❌ Sistema enhanced precisa de correções")
        return False

def main():
    """Função principal"""
    try:
        return run_validation_suite()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Validação interrompida pelo usuário")
        return False
    except Exception as e:
        logger.error(f"❌ Erro crítico durante validação: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)